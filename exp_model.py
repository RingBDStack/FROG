import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Dict, List, Optional
from torch_geometric.nn import HeteroConv, LayerNorm, PositionalEncoding, SAGEConv
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import coalesce
from utils import *
from hop_mode import CoOccurrenceConv, CompletionWithCenterConv, RelJudgeDiff, FDSubspace
import random

class HeteroGraphSAGE(torch.nn.Module):
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        two_hop_relations: set,
        channels: int,
        aggr: str = "sum",
        num_layers: int = 2,
        MAE: float = 0.99,
        dropout: float = 0.0,
        optim: str = 'both',
        fix: bool = False,
        fix_attn: float = 0.5
    ):
        super().__init__()
        self.num_layers = num_layers
        self.node_types = node_types
        self.edge_types = edge_types
        self.two_hop_relations, self.two_hop_neighbors = two_hop_relations
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.optim = 'both'
        
        self.convs = torch.nn.ModuleList()
        self.edge_convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((channels, channels), channels, aggr=aggr)
                    for edge_type in edge_types
                },
                aggr=None,
            )
            self.convs.append(conv)

            edge_conv = HeteroConv(  # node as edge: aggregate 2-hop neighbors
                {
                    edge_type: CoOccurrenceConv(channels, channels, aggr=aggr) 
                    if hop_type == 'co-occurrence' 
                    else CompletionWithCenterConv(channels, channels, aggr=aggr)
                    for edge_type, hop_type in self.two_hop_relations
                },
                aggr=None,
            )
            self.edge_convs.append(edge_conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

        self.gates = torch.nn.ModuleDict()
        for rel, _ in self.two_hop_relations: 
            self.gates[tuple_2_str(rel)] = torch.nn.Sequential(
                torch.nn.Linear(channels * 2, channels),
                torch.nn.ReLU(),
                torch.nn.Linear(channels, 1),
                torch.nn.Sigmoid(),
            )

        self.MAE = MAE

        self.fix = fix
        if self.fix: 
            if fix_attn >= -1e-9:
                self.attn = {tuple_2_str(key): fix_attn for key, _ in self.two_hop_relations}
            else:
                self.attn = {tuple_2_str(key): random.random() for key, _ in self.two_hop_relations}
                
        else:
            self.attn = {tuple_2_str(key): 0.5 for key, _ in self.two_hop_relations}

        self.dropout = nn.Dropout(p=dropout)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[NodeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        edge_index_2hop_dict = self.get_2hop_neighbor_linear_optimized(edge_index_dict)
        
        for layer in range(self.num_layers):
            node_conv = self.convs[layer]
            edge_conv = self.edge_convs[layer]
            norm_dict = self.norms[layer]
            tmp_x_dict, tmp_edge_dict = {}, {}
            for rel, conv in node_conv.convs.items():
                src, edge, dst = rel
                edge_index = edge_index_dict[rel]  # 1-hop neighbors, e.g.: (results, f2p_racesID, races)
                out = conv((x_dict[src], x_dict[dst]), edge_index)
                tmp_x_dict[rel] = out
            for rel, conv in edge_conv.convs.items():
                src, mid, dst = rel
                edge_index = edge_index_2hop_dict[rel]  # 2-hop neighbors, e.g.: (results, races, circuits)
                if edge_index[1].numel() == 0:
                    continue
                out = conv((x_dict[src], x_dict[dst]), x_dict[mid], edge_index)
                tmp_edge_dict[rel] = out
            
            x_out = {}
            for key_node, x_node in tmp_x_dict.items():   # attention aggregation: wheather to use 1-hop(row as node) or 2-hop(row as edge) info
                (src1, mid1, dst1) = key_node
                flag = 0
                if dst1 not in x_out:
                    x_out[dst1] = torch.zeros_like(x_node)
                for key_edge, x_edge in tmp_edge_dict.items():
                    (src2, mid2, dst2) = key_edge
                    if dst1 == dst2 and src1 == mid2:
                        flag = 1
                        gate_input = torch.cat([x_node, x_edge], dim=-1)
                        
                        if self.training and not self.fix:
                            gate = self.gates[tuple_2_str(key_edge)](gate_input)
                            gate = self.gates[tuple_2_str(key_edge)](gate_input)
                            gate = (1 - self.MAE) * gate + self.MAE * self.attn[tuple_2_str(key_edge)]
                            self.attn[tuple_2_str(key_edge)] = gate.detach().clone().mean()
                        else:
                            gate = self.attn[tuple_2_str(key_edge)]

                        if self.optim == 'both':
                            x_out[dst1] = x_out[dst1] + (gate * x_edge + (1 - gate) * x_node)
                        else:
                            x_out[dst1] = x_out[dst1] + (gate * x_edge + (1 - gate) * x_node.detach().clone())

                if flag == 0:
                    x_out[dst1] = x_out[dst1] + x_node
                
            for key in x_dict.keys():
                if key not in x_out.keys():
                    x_out[key] = torch.zeros_like(x_dict[key])
                x_out[key] = x_out[key] + x_dict[key]   # residual connection

            x_dict = {key: norm_dict[key](x) for key, x in x_out.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        return x_dict

    def get_2hop_neighbor_linear_optimized(self, edge_index_dict):
        def intersect_with_indices(a, b):
            a_sorted, idx_a_sorted = torch.sort(a)
            b_sorted, idx_b_sorted = torch.sort(b)

            pos = torch.searchsorted(b_sorted, a_sorted)

            valid = pos < b_sorted.numel()

            pos_valid = pos[valid]
            a_valid = a_sorted[valid]

            equal = b_sorted[pos_valid] == a_valid

            final = valid.clone()
            final[valid] = equal

            common = a_sorted[final]
            idx_a = idx_a_sorted[final]
            idx_b = idx_b_sorted[pos[final]]

            return common, idx_a, idx_b
        
        two_hop_neighbors = {}
        self.edge_search_dict = {(k[0], k[2]): k for k in self.edge_types}

        for (src, mid, dst), hop_type in self.two_hop_relations:

            if hop_type == 'co-occurrence':
                rel_l = self.edge_search_dict.get((mid, src))
            else:  # completion
                rel_l = self.edge_search_dict.get((src, mid))

            rel_r = self.edge_search_dict.get((mid, dst))

            if rel_l is None or rel_r is None:
                continue

            edge_l = edge_index_dict[rel_l]
            edge_r = edge_index_dict[rel_r]

            if hop_type == 'co-occurrence':
                mid_l, src_nodes = edge_l[0], edge_l[1]
            else:
                src_nodes, mid_l = edge_l[0], edge_l[1]

            mid_r, dst_nodes = edge_r[0], edge_r[1]

            common_mid, idx_l, idx_r = intersect_with_indices(
                mid_l, mid_r
            )

            if common_mid.numel() == 0:
                two_hop_neighbors[(src, mid, dst)] = (
                    torch.empty((2, 0), device=self.device, dtype=torch.long),
                    torch.empty((0,), device=self.device, dtype=torch.long)
                )
                continue

            src_aligned = src_nodes[idx_l]
            dst_aligned = dst_nodes[idx_r]

            edge_index = torch.stack([src_aligned, dst_aligned], dim=0)

            two_hop_neighbors[(src, mid, dst)] = (
                edge_index.to(self.device),
                common_mid.to(self.device)
            )

        return two_hop_neighbors

class FD_Model(torch.nn.Module):
    def __init__(self, edge_types, channels):
        super().__init__()
        self.edge_types = edge_types

        self.FD_rel = torch.nn.ModuleDict()
        self.FD_judge = torch.nn.ModuleDict()

        for rel in edge_types:
            key = tuple_2_str(rel)
            self.FD_rel[key] = FDSubspace(channels, channels // 4)
            self.FD_judge[key] = RelJudgeDiff(channels)
            
    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[NodeType, Tensor],
        perm: str = 'id'
    ) -> Dict[NodeType, Tensor]:
        loss_fd = []
        loss_fd_judge = []
        for rel in self.edge_types:
            src, edge, dst = rel
            key = tuple_2_str(rel)
            if edge_index_dict[rel].numel() == 0:
                continue
            if perm == 'id':
                loss_fd.append(self.FD_rel[key].loss(x_dict[src][edge_index_dict[rel][0]], x_dict[dst][edge_index_dict[rel][1]]))
            elif perm == 'random':
                loss_fd.append(
                    self.FD_rel[key].loss(
                        x_dict[src][edge_index_dict[rel][0]], 
                        x_dict[dst][edge_index_dict[rel][1][random.randint(0, len(edge_index_dict[rel][1]) - 1)]]
                    )
                )
            elif perm == 'mean':
                pass
            loss_fd_judge.append(self.FD_judge[key].loss(x_dict[src][edge_index_dict[rel][0]], x_dict[dst][edge_index_dict[rel][1]]))

        return torch.stack(loss_fd).mean(), torch.stack(loss_fd_judge).mean()