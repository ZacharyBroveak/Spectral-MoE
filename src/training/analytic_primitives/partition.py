# Spectral-MoE: src/training/analytic_primitives/partition.py
# Original code: CMoE (https://github.com/JarvisPei/CMoE, revision 42dfc94), Copyright (c)
# 2025 JarvisPei, MIT License (see LICENSE in this directory). Adapted for Spectral-MoE by
# Zachary Broveak.
#
# CMoE balanced clustering of FFN neurons into experts (construct_experts_k_means).
#
# Inputs are per-neuron activation rates and a [tokens, neurons] 0/1 marker matrix (a neuron
# is marked when it is among a token's 8 strongest). Steps:
# 1. The num_shared_experts * neurons_per_expert most frequently marked neurons form the
#    shared expert.
# 2. The most frequently marked remaining neurons seed one centroid per routed expert, using
#    their marker columns.
# 3. Assignment: L1 distances from every neuron's marker pattern to every centroid, with each
#    centroid repeated neurons_per_expert times, form a square problem that the
#    Jonker-Volgenant algorithm (lap.lapjv) solves exactly, which yields equal-size clusters.
#    CMoE runs a single iteration (max_iters = 1).
# 4. Each cluster's representative is its neuron nearest the updated centroid; analytic.py
#    initializes that expert's router row from it.
# 5. Expert MLPs are sliced from the dense gate, up, and down weights by neuron index, on the
#    CPU.
#
# The assignment matrix is n x n in float64 for n routed neurons (about 0.9 GB for Llama-2-7B
# and 2.8 GB for Qwen2.5-7B) and is solved on the CPU, which is why analytic construction is
# CPU-bound and slows down quickly as the FFN gets wider.

import lap
import numpy as np
import torch
from torch import nn

from .model import LlamaMLP


@torch.no_grad()
def construct_experts_k_means(activation_rates, activation_markers, num_experts, num_shared_experts, gate_dim, gate_proj, up_proj, down_proj):
    hidden_size = activation_rates.shape[0]
    neurons_per_expert = hidden_size // num_experts
    expert_groups = []
    remaining_indices = set(range(hidden_size))
    markers = activation_markers.float()
    # Reserve the most frequently active neurons for the shared expert.
    (_, top_indices) = torch.topk(activation_rates, neurons_per_expert * num_shared_experts)
    shared_expert_indices = top_indices.tolist()
    expert_groups.append(shared_expert_indices)
    remaining_indices -= set(shared_expert_indices)
    remaining_indices_list = list(remaining_indices)
    n_samples = len(remaining_indices_list)
    k = num_experts - num_shared_experts
    cluster_size = neurons_per_expert
    # Initialize centroids from the most active remaining neurons.
    remaining_rates = activation_rates[remaining_indices_list]
    (_, top_indices) = torch.topk(remaining_rates, k)
    selected_index = [remaining_indices_list[idx] for idx in top_indices]
    centroids = markers[:, selected_index].clone()
    max_iters = 1
    prev_assignments = None
    for _ in range(max_iters):
        distances = torch.cdist(markers[:, remaining_indices_list].T, centroids.T, p=1)
        distances_np = distances.numpy()
        # Give each centroid an equal number of slots in the assignment problem.
        repeated_distances = np.zeros((n_samples, n_samples))
        for i in range(k):
            repeated_distances[:, i * cluster_size:(i + 1) * cluster_size] = distances_np[:, i:i + 1]
        col_ind = lap.lapjv(repeated_distances)[1]
        assignments = torch.tensor(col_ind // cluster_size)
        if prev_assignments is not None and torch.all(assignments == prev_assignments):
            break
        prev_assignments = assignments.clone()
        for i in range(k):
            cluster_points = markers[:, remaining_indices_list][:, assignments == i]
            if cluster_points.size(1) > 0:
                centroids[:, i] = cluster_points.mean(dim=1)
    representative_indices = []
    for i in range(k):
        cluster_mask = assignments == i
        cluster_points = markers[:, remaining_indices_list][:, cluster_mask]
        cluster_indices = torch.where(cluster_mask)[0]
        # Choose the neuron closest to the updated centroid as its representative.
        distances = torch.cdist(centroids[:, i:i + 1].T, cluster_points.T, p=1)
        closest_idx_in_cluster = torch.argmin(distances)
        original_idx = cluster_indices[closest_idx_in_cluster]
        ori_ori_idx = remaining_indices_list[original_idx]
        representative_indices.append(ori_ori_idx)
        group_indices = [remaining_indices_list[idx] for idx in cluster_indices]
        expert_groups.append(group_indices)
    experts = nn.ModuleList()
    if gate_dim is not None and up_proj is not None and (down_proj is not None):
        for expert_indices in expert_groups:
            expert_mlp = LlamaMLP(gate_dim, len(expert_indices)).to('cpu')
            # Slice matching input rows and output columns for each expert.
            with torch.no_grad():
                expert_mlp.gate_proj.weight.data = gate_proj.weight.data[expert_indices, :]
                expert_mlp.up_proj.weight.data = up_proj.weight.data[expert_indices, :]
                expert_mlp.down_proj.weight.data = down_proj.weight.data[:, expert_indices]
            experts.append(expert_mlp)
    return (expert_groups, experts, representative_indices)
