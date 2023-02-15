import torch
import cxgnncomp as cxgc
import cxgnncomp_backend
import time


def prepare_data():
    dset = "products"
    infeat = 512
    num_head = 1
    # x, ptr, idx, b = cxgc.prepare_data_full_graph(
    #     dset,
    #     feat_len=infeat,
    #     num_head=num_head,
    # )
    x, ptr, idx, b = cxgc.prepare_data_sampled_graph(dset=dset,
                                                     feat_len=infeat,
                                                     num_head=num_head,
                                                     num_seeds=1000)
    return x, ptr, idx, b, num_head


def test_spmm_matmul():
    x, ptr, idx, b, num_head = prepare_data()
    num_rel = 15
    num_center = ptr.shape[0] - 1
    num_edge = idx.shape[0]
    rel = torch.randint(0,
                        num_rel, [idx.shape[0]],
                        dtype=torch.int32,
                        device=x.device)
    rel_int64 = rel.to(torch.int64)
    single_weight = torch.randn([x.shape[-1], x.shape[-1]],
                                dtype=torch.float32,
                                device=x.device)
    dst = torch.repeat_interleave(
        torch.arange(ptr.shape[0] - 1, device=x.device), ptr[1:] - ptr[:-1])
    x_edge = x[idx]
    count = torch.bincount(rel, ).cpu()
    print(count)

    # weights = torch.repeat_interleave(single_weight.unsqueeze(0),
    #                                   num_rel,
    #                                   dim=0).reshape(num_rel, x.shape[-1],
    #                                                  x.shape[-1])
    weights = torch.randn([num_rel, x.shape[-1], x.shape[-1]],
                          dtype=torch.float32,
                          device=x.device)

    # method 2
    output_dst1 = cxgc.TypedLinearS2DMMAggrOP.apply(x, weights, ptr, idx, rel,
                                                    ptr.shape[0] - 1)
    output_dst2 = cxgc.TypedLinearS2DAggrMMOP.apply(x, weights, ptr, idx, rel,
                                                    ptr.shape[0] - 1)
    output_dst3 = cxgc.TypedLinearS2DSort(x, weights, ptr, idx, rel,
                                          ptr.shape[0] - 1)
    print(
        "correct rate output_dst1 vs output_dst2:",
        torch.sum(torch.isclose(output_dst1, output_dst2, atol=1e-2,
                                rtol=1e-2)) / torch.numel(output_dst1))
    print(
        "correct rate output_dst1 vs output_dst3:",
        torch.sum(torch.isclose(output_dst1, output_dst3, atol=1e-2,
                                rtol=1e-2)) / torch.numel(output_dst1))
    output_edge1 = cxgc.TypedLinearE2EOP.apply(x_edge, weights, rel_int64)
    output_edge2 = cxgc.TypedLinearE2EOP.apply(x_edge, weights, rel_int64,
                                               False, count)
    output_edge3 = cxgc.TypedLinearS2EOP.apply(x, weights, rel_int64, idx)
    output_edge4 = cxgc.TypedLinearS2EOP.apply(x, weights, rel_int64, idx,
                                               False, count)
    print(
        "correct rate output_edge1 vs output_edge2:",
        torch.sum(
            torch.isclose(output_edge1, output_edge2, atol=1e-2, rtol=1e-2)) /
        torch.numel(output_edge1))
    print(
        "correct rate output_edge1 vs output_edge3:",
        torch.sum(
            torch.isclose(output_edge1 / (output_edge1[0] / output_edge3[0]),
                          output_edge3,
                          atol=1e-2,
                          rtol=1e-2)) / torch.numel(output_edge1))
    # print(output_edge1, output_edge3)
    print(
        "correct rate output_edge1 vs output_edge4:",
        torch.sum(
            torch.isclose(output_edge1, output_edge4, atol=1e-2, rtol=1e-2)) /
        torch.numel(output_edge1))

    cxgc.prof(
        "typed linear",
        "s2d aggr mm", lambda: cxgc.TypedLinearS2DMMAggrOP.apply(
            x, weights, ptr, idx, rel, ptr.shape[0] - 1))
    cxgc.prof(
        "typed linear",
        "s2d mm aggr", lambda: cxgc.TypedLinearS2DAggrMMOP.apply(
            x, weights, ptr, idx, rel, ptr.shape[0] - 1))
    cxgc.prof(
        "typed linear", "s2d sort", lambda: cxgc.TypedLinearS2DSort(
            x, weights, ptr, idx, rel, ptr.shape[0] - 1))

    cxgc.prof("typed linear", "e2e",
              lambda: cxgc.TypedLinearE2EOP.apply(x_edge, weights, rel_int64))
    cxgc.prof(
        "typed linear", "e2e with count", lambda: cxgc.TypedLinearE2EOP.apply(
            x_edge, weights, rel_int64, False, count))
    cxgc.prof("typed linear", "s2e",
              lambda: cxgc.TypedLinearS2EOP.apply(x, weights, rel_int64, idx))
    cxgc.prof(
        "typed linear", "s2e with count", lambda: cxgc.TypedLinearS2EOP.apply(
            x, weights, rel_int64, idx, False, count))

    new_idx = torch.arange(0, idx.shape[0], device=idx.device)
    cxgc.prof(
        "aggregation", "e2d", lambda: cxgnncomp_backend.sage_sum_forward(
            output_edge1, ptr, new_idx, num_center))

    cxgc.prof(
        "aggregation", "s2d",
        lambda: cxgnncomp_backend.sage_sum_forward(x, ptr, idx, num_center))

    # cxgc.tune_spmm(ptr.shape[0] - 1, idx.shape[0], x.shape[1],
    #                cxgnncomp_backend.run_spmm_configurable,
    #                [ptr, idx, x, ptr.shape[0] - 1])

    # cxgc.tune_spmm(ptr.shape[0] - 1, idx.shape[0], x.shape[1],
    #                cxgnncomp_backend.run_spmm_configurable,
    #                [ptr, new_idx, output_edge1, ptr.shape[0] - 1])

    output_dst4 = torch.zeros([num_center, x.shape[-1]],
                              dtype=x.dtype,
                              device=x.device)
    output_dst4.index_add_(0, dst, output_edge1)
    print(
        "correct rate output_dst1 vs output_dst4:",
        torch.sum(torch.isclose(output_dst1, output_dst4, atol=1e-2,
                                rtol=1e-2)) / torch.numel(output_dst1))
    cxgc.prof("aggregation", "e2d index_add_",
              lambda: output_dst4.index_add_(0, dst, output_edge1))


if __name__ == "__main__":
    test_spmm_matmul()