import torch
import time
from torch.nn import Parameter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot
from cxgnncomp_backend import (
    edge_attention,
    sage_sum_forward_edge_value,
    gather,
    sage_sum_forward,
    aggr_rel,
    sage_mean_forward,
    selective_aggr,
    selective_aggr_bwd,
    aggr_rgcn_direct_func,
)
from .util import log
import torch.nn.functional as F
from torch_scatter import segment_csr, gather_csr, scatter
from .timer import TimerOP
from .graph_kernel import SpMMValOP
from .typed_linear import TypedLinearE2EOP, TypedLinearS2DPushOP
from .util import global_tuner
import cxgnncomp_backend

torch.fx.wrap("edge_attention")
torch.fx.wrap("sage_sum_forward_edge_value")
torch.fx.wrap("gather")
torch.fx.wrap("sage_sum_forward")
torch.fx.wrap("aggr_rel")
torch.fx.wrap("sage_mean_forward")


class AggrOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, ptr, idx, num_center):
        # torch.cuda.synchronize()
        ctx.save_for_backward(x, ptr, idx, num_center)
        # torch.cuda.synchronize()
        output_dst = global_tuner.tune_graph(
            num_center,
            idx.shape[0],
            x.shape[-1],
            cxgnncomp_backend.run_spmm_configurable,
            [
                ptr,
                idx,
                x,
                num_center,
            ],
        )
        # torch.cuda.synchronize()
        return output_dst

    @staticmethod
    def backward(ctx, grad_out):
        x, ptr, idx, num_center = ctx.saved_tensors
        # torch.cuda.synchronize()

        # # TODO: now assuming asymmetric graph
        # assert num_center == x.shape[0] and x.shape == grad_out.shape
        torch.cuda.synchronize()
        grad_vin = global_tuner.tune_graph(
            num_center,
            idx.shape[0],
            grad_out.shape[-1],
            cxgnncomp_backend.run_spmm_configurable,
            [ptr, idx, grad_out, num_center],
        )

        # grad_vin = global_tuner.tune_graph(
        #     num_center, idx.shape[0], grad_out.shape[-1],
        #     cxgnncomp_backend.run_spmm_configurable_bwd,
        #     [ptr, idx, x, grad_out, num_center])
        torch.cuda.synchronize()
        return grad_vin, None, None, None


class AggrDirectedOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, ptr, idx, num_center):
        # torch.cuda.synchronize()
        ctx.save_for_backward(x, ptr, idx, num_center)
        # torch.cuda.synchronize()
        output_dst = global_tuner.tune_graph(
            num_center,
            idx.shape[0],
            x.shape[-1],
            cxgnncomp_backend.run_spmm_configurable,
            [
                ptr,
                idx,
                x,
                num_center,
            ],
        )
        # torch.cuda.synchronize()
        return output_dst

    @staticmethod
    def backward(ctx, grad_out):
        x, ptr, idx, num_center = ctx.saved_tensors
        # torch.cuda.synchronize()
        grad_vin = global_tuner.tune_graph(
            num_center,
            idx.shape[0],
            grad_out.shape[-1],
            cxgnncomp_backend.run_spmm_configurable_bwd,
            [ptr, idx, x, grad_out, num_center],
        )
        torch.cuda.synchronize()
        return grad_vin, None, None, None


class MySageConv(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        root_weight: bool = True,
        bias: bool = True,
    ):
        super(MySageConv, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.root_weight = root_weight
        self.lin_l = torch.nn.Linear(in_channels, hidden_channels, bias=bias)
        if self.root_weight:
            self.lin_r = torch.nn.Linear(in_channels, hidden_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x, ptr, idx, num_node):

        if self.in_channels > self.hidden_channels:
            out = self.lin_l(x)
            # out = AggrOP.apply(out, ptr, idx, num_node)
            out = sage_mean_forward(out, ptr, idx, num_node)
        else:
            # out = AggrOP.apply(x, ptr, idx, num_node)
            # print(x.device, ptr.device, self.lin_l.weight.device, torch.max(idx[:ptr[num_node]]), x.shape, ptr.shape, ptr[num_node], idx.shape, num_node)
            # torch.cuda.synchronize()
            out = sage_mean_forward(x, ptr, idx, num_node)
            # torch.cuda.synchronize()
            out = self.lin_l(out)

        if self.root_weight:
            if num_node != 0:
                out += self.lin_r(x[:num_node])
            else:
                out += self.lin_r(x)
        return out


class MyGINConv(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels):
        super(MyGINConv, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.nn = torch.nn.Linear(in_channels, hidden_channels, bias=False)
        self.init_eps = 0.0
        self.eps = torch.nn.Parameter(torch.Tensor([self.init_eps]))
        self.reset_parameters()

    def reset_parameters(self):
        self.nn.reset_parameters()
        self.eps.data.fill_(self.init_eps)

    def forward(self, x, ptr, idx, num_node):
        out = sage_mean_forward(x, ptr, idx, num_node)
        out += (1 + self.eps) * x[:num_node]
        out = self.nn(out)
        return out


class SeastarRGCNOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weights, ptr, idx, rel, num_center):
        # weights = torch.transpose(weights, 1, 2)
        rel = rel.to(torch.int64)
        ctx.save_for_backward(x, weights, ptr, idx, rel)
        out = cxgnncomp_backend.seastar_forward(x, ptr, idx, weights, rel)
        # torch.cuda.synchronize()
        # print("out", out, x, weights)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weights, ptr, idx, rel = ctx.saved_tensors
        grad_x, grad_weight = cxgnncomp_backend.seastar_backward(
            x, ptr, idx, weights, rel, grad_out
        )
        return grad_x, grad_weight, None, None, None, None


class RGCNOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weights, ptr, idx, rel, num_center):
        ctx.save_for_backward(x, weights, ptr, idx, rel)
        num_rel = weights.shape[0]
        output = torch.zeros([num_center, weights.shape[-1]], device=x.device)
        for i in range(num_rel):
            weights[i] = TimerOP.apply(weights[i], f"mm{i}", True)
            transformed_x = torch.mm(x, weights[i])
            transformed_x = TimerOP.apply(transformed_x, f"mm{i}", False)
            transformed_x = TimerOP.apply(transformed_x, f"selective{i}", True)
            selective_aggr(transformed_x, ptr, idx, (rel == i), output, num_center)
            output = TimerOP.apply(output, f"selective{i}", False)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        x, weights, ptr, idx, rel = ctx.saved_tensors
        num_rel = weights.shape[0]
        grad_x = torch.zeros_like(x)
        grad_weights = []
        num_center = grad_out.shape[0]
        num_node = x.shape[0]
        x_t = x.transpose(0, 1)
        for i in range(num_rel):
            grad_selective = torch.zeros(
                [num_node, grad_out.shape[-1]], device=x.device
            )
            grad_selective = TimerOP.apply(grad_selective, f"selective_bwd{i}", True)
            selective_aggr_bwd(
                grad_out, ptr, idx, (rel == i), grad_selective, num_center
            )  # pass grad through selective_aggr
            grad_selective = TimerOP.apply(grad_selective, f"selective_bwd{i}", False)
            grad_selective = TimerOP.apply(grad_selective, f"mm_bwd{i}", True)
            grad_x += torch.mm(grad_selective, weights[i].transpose(0, 1))
            grad_selective = TimerOP.apply(grad_selective, f"mm_bwd{i}", False)
            grad_weights.append(torch.mm(x_t, grad_selective))
        return grad_x, torch.stack(grad_weights), None, None, None, None


def RGCNOP_sorted(x, weights, src, dst, num_feat_per_rel, num_center):
    num_rel = weights.shape[0]
    output = torch.zeros([num_center, weights.shape[-1]], device=x.device)
    cnt = 0
    for i in range(num_rel):
        s = src[cnt : cnt + num_feat_per_rel[i]]
        d = dst[cnt : cnt + num_feat_per_rel[i]]
        cnt += num_feat_per_rel[i]
        feat = x[s]
        transformed_feat = F.linear(feat, weights[i].T)
        output.index_add_(0, d, transformed_feat)
    return output


class RGCNOP2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weights, ptr, idx, rel, num_center):
        num_rel = weights.shape[0]
        output = torch.zeros([num_center, weights.shape[-1]], device=x.device)
        aggr_outputs = []
        for i in range(num_rel):
            aggr_output = torch.zeros([num_center, weights.shape[-2]], device=x.device)
            selective_aggr(x, ptr, idx, (rel == i), aggr_output, num_center)
            output += torch.mm(aggr_output, weights[i])
            # output += torch.empty([num_center, weights.shape[-1]],
            #                       device=x.device)
            aggr_outputs.append(aggr_output)
        ctx.save_for_backward(x, weights, ptr, idx, rel, torch.stack(aggr_outputs))
        return output

    @staticmethod
    def backward(ctx, grad_out):
        x, weights, ptr, idx, rel, aggr_outputs = ctx.saved_tensors
        num_rel = weights.shape[0]
        grad_x = torch.zeros_like(x)
        grad_weights = []
        num_center = grad_out.shape[0]
        for i in range(num_rel):
            grad_mm = torch.mm(grad_out, weights[i].transpose(0, 1))
            # grad_mm = torch.empty([num_center, weights.shape[-2]],
            #                       device=x.device)
            grad_weights.append(torch.mm(aggr_outputs[i].transpose(0, 1), grad_out))
            selective_aggr_bwd(
                grad_mm, ptr, idx, (rel == i), grad_x, num_center
            )  # pass grad through selective_aggr
        return grad_x, torch.stack(grad_weights), None, None, None, None


class RGCNOP3(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weights, ptr, idx, rel, num_center):
        ctx.save_for_backward(x, weights, ptr, idx, rel)
        output = aggr_rgcn_direct_func(x, ptr, idx, weights, rel.int(), num_center)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        x, weights, ptr, idx, rel = ctx.saved_tensors
        return torch.randn_like(x), torch.randn_like(weights), None, None, None, None


def run_rgcn4(x, ptr, idx, edge_types, linear, num_center, num_edge, preproccessed):
    # print(x.shape, ptr.shape, idx.shape, edge_types.shape, linear.shape,
    #       x.dtype, ptr.dtype, idx.dtype, edge_types.dtype, linear.dtype,
    #       x.device, ptr.device, idx.device, edge_types.device, linear.device)
    # impl 1
    torch.cuda.synchronize()
    t0 = time.time()
    x_idxed = x[idx[:num_edge]]
    torch.cuda.synchronize()
    t1 = time.time()
    print("index time", t1 - t0)
    out = TypedLinearE2EOP.apply(x_idxed, linear, edge_types, preproccessed)
    torch.cuda.synchronize()
    t2 = time.time()
    print("typed mm", t2 - t1)
    new_idx = torch.arange(0, num_edge, device=x_idxed.device)
    out = AggrDirectedOP.apply(out, ptr, new_idx, num_center)
    return out


class MyRGCNConv(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, num_rel):
        super(MyRGCNConv, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_rel = num_rel
        assert num_rel > 0
        self.linear = torch.nn.Parameter(
            torch.randn(num_rel, in_channels, hidden_channels)
        )
        log.info("linear shape: {}".format(self.linear.shape))
        self.register_parameter("rel_weight", self.linear)
        self.single_linear = torch.nn.Linear(in_channels, hidden_channels)
        self.reset_parameters()
        self.preprocessed = []

    def reset_parameters(self):
        glorot(self.linear)
        # self.linear.reset_parameters()
        self.single_linear.reset_parameters()
        pass

    def forward(self, x, ptr, idx, edge_types, num_node):
        if idx.shape[0] > 1e8:
            out = RGCNOP.apply(x, self.linear, ptr, idx, edge_types, num_node)
            return out

        if len(self.preprocessed) == 0:
            self.preprocessed = TypedLinearE2EOP.preprocess(self.linear, edge_types)
        out = run_rgcn4(
            x,
            ptr,
            idx,
            edge_types,
            self.linear,
            num_node,
            num_edge=edge_types.shape[0],
            preproccessed=self.preprocessed,
        )

        deg = ptr[1:] - ptr[:-1]
        # out = SeastarRGCNOP.apply(x, self.linear, ptr, idx, edge_types,
        #                           num_node)
        # torch.cuda.synchronize()

        out = out / deg.unsqueeze(-1)[: out.shape[0]]
        # out = self.single_linear(x)
        # out = sage_mean_forward(out, ptr, idx, num_node)
        return out


class MyGATConv(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = False,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
        layer=-1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # if out_channels % heads != 0:
        #     heads = 1
        self.heads = heads
        assert heads > 0
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.layer = layer

        self.lin_src = Parameter(torch.Tensor(heads * out_channels, in_channels))

        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.edge_softmax_schedule = "not_fused"
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_src)
        glorot(self.att_src)
        glorot(self.att_dst)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def edge_softmax_fused(self, ptr, idx, att_src, att_dst, num_edge, relu_l):
        return edge_attention(
            ptr=ptr,
            idx=idx,
            att_src=att_src,
            att_dst=att_dst,
            num_edge=num_edge,
            relu_l=relu_l,
        )

    def edge_softmax_opwise(self, ptr, idx, att_src, att_dst, num_edge, relu_l):
        alpha_src = torch.index_select(att_src, 0, idx[:num_edge])
        alpha_dst = gather_csr(att_dst, ptr)
        alpha = F.leaky_relu(alpha_src + alpha_dst, relu_l)
        with torch.no_grad():
            alpha_max = segment_csr(alpha, ptr, reduce="max")
            alpha_max = gather_csr(alpha_max, ptr)
        alpha = torch.exp(alpha - alpha_max)
        out_sum = segment_csr(alpha, ptr, reduce="sum") + 1e-16
        out_sum = gather_csr(out_sum, ptr)
        edge_value = alpha / out_sum
        return edge_value

    def edge_softmax(self, ptr, idx, att_src, att_dst, num_edge, relu_l):
        if self.edge_softmax_schedule == "fused":
            return self.edge_softmax_fused(ptr, idx, att_src, att_dst, num_edge, relu_l)
        else:
            return self.edge_softmax_opwise(
                ptr, idx, att_src, att_dst, num_edge, relu_l
            )

    def forward_many(self, x, ptr, idx, num_dst, num_src, num_edge):
        H, C = self.heads, self.out_channels
        assert x.dim() == 2
        # x_src = x_dst = torch.mm(x[:num_src], self.lin_src).view(-1, H, C)
        x = TimerOP.apply(x, "linear1", True)
        x_src = x_dst = F.linear(x[:num_src], self.lin_src).view(-1, H, C)
        x_src = TimerOP.apply(x_src, "linear1", False)
        x_src = TimerOP.apply(x_src, "sum1", True)
        alpha_src = (x_src * self.att_src).sum(dim=-1).view(-1, H)
        alpha_dst = (x_dst[:num_dst] * self.att_dst).sum(dim=-1).view(-1, H)
        alpha_dst = TimerOP.apply(alpha_dst, "sum1", False)
        alpha_dst = TimerOP.apply(alpha_dst, "softmax1", True)
        edge_value = self.edge_softmax(
            ptr=ptr,
            idx=idx,
            att_src=alpha_src,
            att_dst=alpha_dst,
            num_edge=num_edge,
            relu_l=self.negative_slope,
        )
        edge_value = TimerOP.apply(edge_value, "softmax1", False)
        edge_value = TimerOP.apply(edge_value, "aggregation", True)
        # out = sage_sum_forward_edge_value(x_src, ptr, idx, edge_value, num_dst)
        if self.heads == 1:
            out = sage_sum_forward_edge_value(
                x_src, ptr, idx, edge_value.squeeze(), num_dst
            )
        else:
            out = SpMMValOP.apply(x_src, ptr, idx, edge_value, num_dst)
        out = TimerOP.apply(out, "aggregation", False)
        if self.concat:
            out = out.view(-1, H * C)
        elif out.shape[1] == self.heads:
            out = out.mean(dim=1)  # NOTE: requires out to be [-1, H, C]
        # else: already sumed/averaged
        if self.bias is not None:
            return out + self.bias
        else:
            return out

    def forward_1(self, x, ptr, idx, num_dst, num_src, num_edge):
        x = TimerOP.apply(x, "einsum1", True)
        alpha_src = torch.einsum(
            "mn,nho,ho->mh",
            x,
            self.lin_src.T.view(-1, self.heads, self.out_channels),
            self.att_src.squeeze(0),
        ).view(-1, self.heads)
        alpha_dst = torch.einsum(
            "mn,nho,ho->mh",
            x[:num_dst],
            self.lin_src.T.view(-1, self.heads, self.out_channels),
            self.att_dst.squeeze(0),
        ).view(-1, self.heads)
        alpha_dst = TimerOP.apply(alpha_dst, "einsum1", False)
        alpha_dst = TimerOP.apply(alpha_dst, "softmax1", True)
        edge_value = self.edge_softmax(
            ptr=ptr,
            idx=idx,
            att_src=alpha_src,
            att_dst=alpha_dst,
            num_edge=num_edge,
            relu_l=self.negative_slope,
        )
        edge_value = TimerOP.apply(edge_value, "softmax1", False)
        if self.out_channels < self.in_channels:
            x = TimerOP.apply(x, "mm1", True)
            transformed = torch.mm(x, self.lin_src.T).view(
                -1, self.heads, self.out_channels
            )
            transformed = TimerOP.apply(transformed, "mm1", False)
            transformed = TimerOP.apply(transformed, "aggregation", True)
            x = sage_sum_forward_edge_value(
                transformed, ptr, idx, edge_value, num_dst
            ).squeeze()
            # x = AggrOP.apply(transformed.squeeze(), ptr, idx, num_dst).squeeze()
            x = TimerOP.apply(x, "aggregation", False)
        else:
            x = TimerOP.apply(x, "aggregation", True)
            x = sage_sum_forward_edge_value(x, ptr, idx, edge_value.squeeze(), num_dst)
            # x = AggrOP.apply(x.squeeze(), ptr, idx, num_dst).squeeze()
            x = TimerOP.apply(x, "aggregation", False)
            x = TimerOP.apply(x, "mm1", True)
            x = torch.mm(x, self.lin_src.T)
            x = TimerOP.apply(x, "mm1", False)
        if self.bias is not None:
            # x += self.bias
            return x + self.bias
        else:
            return x

    def forward(self, x, ptr, idx, num_dst, num_src, num_edge):
        ptr = ptr[:num_dst + 1]
        idx = idx[:num_edge]
        if self.heads == 1:
            return self.forward_1(x, ptr, idx, num_dst, num_src, num_edge)
        else:
            return self.forward_many(x, ptr, idx, num_dst, num_src, num_edge)


class MyGCNConv(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalize: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin = Linear(
            in_channels, out_channels, bias=False, weight_initializer="glorot"
        )  # for consistency with PyG
        # self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.bias is not None:
            self.bias.data.fill_(0)

    def gcn_norm(self, ptr: torch.Tensor, idx: torch.Tensor):
        deg = ptr[1:] - ptr[:-1]  # in degree
        # deg_from = idx.bincount()
        # deg_from = deg_from.index_select(0, idx)
        deg_to = deg.repeat_interleave(deg)
        # assert(deg_to.shape == deg_from.shape)
        # edge_value = (deg_to * deg_from).pow(-1/2)
        edge_value = (deg_to.float()).pow(-1)
        edge_value.masked_fill_(edge_value == float("inf"), 0.0)
        return edge_value

    def forward(self, x, ptr, idx, num_node):
        x = TimerOP.apply(x, "linear", True)
        mm_before = self.in_channels > self.out_channels
        # mm_before = False
        if mm_before:
            out = self.lin(x)
        else:
            out = x
        out = TimerOP.apply(out, "linear", False)
        if self.normalize:
            out = TimerOP.apply(out, "aggregation", True)
            TimerOP.apply(out, "gcn norm", True)
            edge_value = self.gcn_norm(ptr, idx)
            TimerOP.apply(edge_value, "gcn norm", False)
            # print(out.shape, ptr.shape, idx.shape, edge_value.shape, num_node)
            out = sage_sum_forward_edge_value(out, ptr, idx, edge_value, num_node)
            out = TimerOP.apply(out, "aggregation", False)
        else:
            # out = sage_sum_forward(out, ptr, idx, num_node)
            out = AggrOP.apply(out, ptr, idx, num_node)

        if not mm_before:
            out = self.lin(out)
        # print("model out", out.reshape(-1)[:10])
        if self.bias is not None:
            return out + self.bias
        else:
            return out
