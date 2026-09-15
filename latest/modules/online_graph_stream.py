"""
modules/online_graph_stream.py
==============================
PHASE 7: Online Streaming Bipartite Graph Maintenance & GNN State Tracker

Maintains a sliding-window temporal bipartite graph (V_host, V_service, E) in memory.
Continuously updates unsupervised node features and updates GRU temporal state vectors (h_{t-1} -> h_t)
in real-time as new traffic flows arrive.
"""

import math
import torch
import numpy as np
from modules.bipartite_graph_builder import BipartiteGraphBuilder
from modules.dynamic_temporal_gnn import DynamicBipartiteTemporalGNN


class OnlineGraphStream:
    """Sliding-Window Online Bipartite Graph & GNN State Tracker."""

    def __init__(self, gnn_model, scalers, window_size=500):
        self.gnn_model = gnn_model
        self.host_scaler = scalers["host_scaler"]
        self.service_scaler = scalers["service_scaler"]
        self.edge_scaler = scalers["edge_scaler"]
        self.window_size = window_size

        self.builder = BipartiteGraphBuilder()
        self.flow_buffer = []

        self.prev_host_h = None
        self.prev_service_h = None

    def ingest_flow(self, flow_rec):
        """Ingests a single flow record into the real-time graph stream."""
        self.flow_buffer.append(flow_rec)
        if len(self.flow_buffer) > self.window_size:
            self.flow_buffer.pop(0)

        self.builder.add_flow(
            src_ip=flow_rec["src_ip"],
            dst_ip=flow_rec["dst_ip"],
            dst_port=flow_rec["dst_port"],
            protocol=flow_rec["protocol"],
            timestamp=flow_rec["timestamp"],
            byte_count=flow_rec["byte_count"],
            pkt_count=flow_rec["pkt_count"],
            duration=flow_rec["duration"],
            xgb_prob=flow_rec["xgb_prob"],
            ae_score=flow_rec["ae_score"],
            fusion_score=flow_rec["fusion_score"],
            attack_label=flow_rec.get("attack_label", 0),
        )

    def evaluate_realtime_gnn_risk(self):
        """
        Executes real-time Dynamic Bipartite Temporal GNN inference on the current online graph.
        Returns edge risk probabilities and updated GRU temporal node states.
        """
        pyg_data, h_map, s_map = self.builder.to_pyg_hetero()

        if ("host", "connects_to", "service") not in pyg_data.edge_types or pyg_data["host", "connects_to", "service"].edge_index.shape[1] == 0:
            return np.empty((0,), dtype=np.float32), pyg_data, h_map, s_map

        hx_numpy = pyg_data["host"].x.numpy()
        sx_numpy = pyg_data["service"].x.numpy()
        eidx_numpy = pyg_data["host", "connects_to", "service"].edge_index.numpy()
        eattr_numpy = pyg_data["host", "connects_to", "service"].edge_attr.numpy()

        hx_scaled = torch.tensor(self.host_scaler.transform(hx_numpy), dtype=torch.float32)
        sx_scaled = torch.tensor(self.service_scaler.transform(sx_numpy), dtype=torch.float32)
        eidx = torch.tensor(eidx_numpy, dtype=torch.long)
        eattr_scaled = torch.tensor(self.edge_scaler.transform(eattr_numpy), dtype=torch.float32)

        self.gnn_model.eval()
        with torch.no_grad():
            logits, next_h_h, next_h_s = self.gnn_model(
                hx_scaled, sx_scaled, eidx, eattr_scaled, self.prev_host_h, self.prev_service_h
            )
            probs = torch.sigmoid(logits).numpy()

        # Update GRU memory state
        self.prev_host_h = next_h_h
        self.prev_service_h = next_h_s

        return probs, pyg_data, h_map, s_map
