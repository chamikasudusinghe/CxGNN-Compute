#!/bin/bash

# This script is used to run the experiments for WiseGraph benchmark for the paper "X"

dsets=(cora pubmed corafull reddit arxiv products)
models=(GAT SAGE GCN GIN)
graph_types=(CSR_Layer)
num_layers=(2)
hidden_feats=(32)

# Figure 16 & 17
for graph_type in ${graph_types[@]}; do
    for dset in ${dsets[@]}; do
        for model in ${models[@]}; do
            for num_layer in ${num_layers[@]}; do
                for hidden_feat in ${hidden_feats[@]}; do
                    echo "python3 test_model.py --dataset ${dset} --model ${model} --graph_type ${graph_type} --hidden_feat ${hidden_feat} --num_layer ${num_layer}"
                    python3 test_model.py --dataset ${dset} --model ${model} --graph_type ${graph_type} --hidden_feat ${hidden_feat} --num_layer ${num_layer}
                done
            done
        done
    done
done

dsets=(reddit)
models=(GCN)
graph_types=(CSR_Layer)
num_layers=(2, 3, 4, 8)
hidden_feats=(32, 64, 128, 256, 512, 1024)

# Figure 18 & 19 (only the first result is relavent for Figure 19)
for graph_type in ${graph_types[@]}; do
    for dset in ${dsets[@]}; do
        for model in ${models[@]}; do
            for num_layer in ${num_layers[@]}; do
                for hidden_feat in ${hidden_feats[@]}; do
                    echo "python3 test_model.py --dataset ${dset} --model ${model} --graph_type ${graph_type} --hidden_feat ${hidden_feat} --num_layer ${num_layer}"
                    python3 test_model.py --dataset ${dset} --model ${model} --graph_type ${graph_type} --hidden_feat ${hidden_feat} --num_layer ${num_layer}
                done
            done
        done
    done
done

dsets=(ogbn-papers100M_1 ogbn-papers100M_2 ogbn-papers100M_5 ogbn-papers100M_10)
models=(GCN)
graph_types=(CSR_Layer)
num_layers=(2)
hidden_feats=(32)

# Table 5
for graph_type in ${graph_types[@]}; do
    for dset in ${dsets[@]}; do
        for model in ${models[@]}; do
            for num_layer in ${num_layers[@]}; do
                for hidden_feat in ${hidden_feats[@]}; do
                    echo "python3 test_model.py --dataset ${dset} --model ${model} --graph_type ${graph_type} --hidden_feat ${hidden_feat} --num_layer ${num_layer}"
                    python3 test_model.py --dataset ${dset} --model ${model} --graph_type ${graph_type} --hidden_feat ${hidden_feat} --num_layer ${num_layer}
                done
            done
        done
    done
done