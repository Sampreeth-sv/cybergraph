
"""
modules/bipartite_graph_builder.py

PHASE 2 & 3: Strict Bipartite Graph Builder

Constructs a bipartite graph:
    HOST nodes -> SERVICE nodes

Each dataset flow is retained as an individual edge record.
A unique flow_id is preserved separately from model features
to support exact alignment of graph predictions with source rows.

Node features:
    HOST:
      [out_degree, total_bytes, total_packets, port_entropy,
       protocol_diversity, avg_xgboost_prob, max_ae_score,
       avg_fusion_score]

    SERVICE:
      [in_degree, total_bytes, total_packets, host_diversity,
       protocol_count, avg_xgboost_prob, max_ae_score,
       avg_fusion_score]

Edge features:
    [timestamp, dst_port, protocol, byte_count, pkt_count,
     duration, xgb_prob, ae_score, fusion_score]

Important:
    - attack_label is retained only as the edge target.
    - flow_id is metadata, NOT a model input feature.
    - No HOST-HOST or SERVICE-SERVICE edges are created.
"""

import math
from collections import defaultdict

import networkx as nx
import numpy as np


HOST_FEATURE_NAMES = [
    "out_degree",
    "total_bytes",
    "total_packets",
    "port_entropy",
    "protocol_diversity",
    "avg_xgboost_prob",
    "max_ae_score",
    "avg_fusion_score",
]

SERVICE_FEATURE_NAMES = [
    "in_degree",
    "total_bytes",
    "total_packets",
    "host_diversity",
    "protocol_count",
    "avg_xgboost_prob",
    "max_ae_score",
    "avg_fusion_score",
]

EDGE_FEATURE_NAMES = [
    "timestamp",
    "dst_port",
    "protocol",
    "byte_count",
    "pkt_count",
    "duration",
    "xgb_prob",
    "ae_score",
    "fusion_score",
]


def _entropy(counts):
    total = sum(counts)

    if total <= 1 or len(counts) <= 1:
        return 0.0

    probabilities = [count / total for count in counts if count > 0]

    if len(probabilities) <= 1:
        return 0.0

    return -sum(
        p * math.log2(p) for p in probabilities
    ) / math.log2(len(probabilities))


class BipartiteGraphBuilder:
    def __init__(self):
        self.g = nx.Graph()

        self.host_nodes = set()
        self.service_nodes = set()

        self._stats = defaultdict(
            lambda: {
                "bytes": 0.0,
                "packets": 0.0,
                "connected_peers": defaultdict(int),
                "protocols": set(),
                "xgb_probs": [],
                "ae_scores": [],
                "fusion_scores": [],
                "flow_count": 0,
            }
        )

        self._seen_flow_ids = set()

    def make_host_id(self, ip_str):
        return f"HOST:{ip_str}"

    def make_service_id(self, port, protocol):
        return f"SERVICE:{int(port)}/{int(protocol)}"

    def add_flow(
        self,
        src_ip,
        dst_ip,
        dst_port,
        protocol,
        timestamp,
        byte_count=0.0,
        pkt_count=0.0,
        duration=0.0,
        xgb_pred=0,
        xgb_prob=0.0,
        ae_score=0.0,
        fusion_score=0.0,
        attack_label=0,
        flow_id=None,
    ):
        if flow_id is None:
            raise ValueError(
                "flow_id is required for reliable flow alignment."
            )

        flow_id = int(flow_id)

        if flow_id < 0:
            raise ValueError(f"Invalid flow_id: {flow_id}")

        if flow_id in self._seen_flow_ids:
            raise ValueError(f"Duplicate flow_id: {flow_id}")

        self._seen_flow_ids.add(flow_id)

        host_node = self.make_host_id(src_ip)
        service_node = self.make_service_id(dst_port, protocol)

        self.host_nodes.add(host_node)
        self.service_nodes.add(service_node)

        self.g.add_node(
            host_node,
            node_type="HOST",
            raw_id=str(src_ip),
        )

        self.g.add_node(
            service_node,
            node_type="SERVICE",
            raw_id=f"{dst_port}/{protocol}",
        )

        edge_data = {
            "flow_id": flow_id,
            "timestamp": float(timestamp),
            "src_ip": str(src_ip),
            "dst_ip": str(dst_ip),
            "dst_port": int(dst_port),
            "protocol": int(protocol),
            "byte_count": float(byte_count),
            "pkt_count": float(pkt_count),
            "duration": float(duration),
            "xgb_pred": int(xgb_pred),
            "xgb_prob": float(xgb_prob),
            "ae_score": float(ae_score),
            "fusion_score": float(fusion_score),
            "attack_label": int(attack_label),
        }

        if self.g.has_edge(host_node, service_node):
            self.g[host_node][service_node]["weight"] += 1
            self.g[host_node][service_node]["flows"].append(edge_data)
        else:
            self.g.add_edge(
                host_node,
                service_node,
                weight=1,
                flows=[edge_data],
            )

        # HOST statistics
        host_stats = self._stats[host_node]

        host_stats["bytes"] += float(byte_count)
        host_stats["packets"] += float(pkt_count)
        host_stats["connected_peers"][service_node] += 1
        host_stats["protocols"].add(int(protocol))
        host_stats["xgb_probs"].append(float(xgb_prob))
        host_stats["ae_scores"].append(float(ae_score))
        host_stats["fusion_scores"].append(float(fusion_score))
        host_stats["flow_count"] += 1

        # SERVICE statistics
        service_stats = self._stats[service_node]

        service_stats["bytes"] += float(byte_count)
        service_stats["packets"] += float(pkt_count)
        service_stats["connected_peers"][host_node] += 1
        service_stats["protocols"].add(int(protocol))
        service_stats["xgb_probs"].append(float(xgb_prob))
        service_stats["ae_scores"].append(float(ae_score))
        service_stats["fusion_scores"].append(float(fusion_score))
        service_stats["flow_count"] += 1

    def compute_host_features(self, host_node):
        stats = self._stats[host_node]

        degree = len(stats["connected_peers"])

        entropy_val = _entropy(
            list(stats["connected_peers"].values())
        )

        avg_xgb = (
            float(np.mean(stats["xgb_probs"]))
            if stats["xgb_probs"] else 0.0
        )

        max_ae = (
            float(np.max(stats["ae_scores"]))
            if stats["ae_scores"] else 0.0
        )

        avg_fusion = (
            float(np.mean(stats["fusion_scores"]))
            if stats["fusion_scores"] else 0.0
        )

        return [
            float(degree),
            float(stats["bytes"]),
            float(stats["packets"]),
            float(entropy_val),
            float(len(stats["protocols"])),
            avg_xgb,
            max_ae,
            avg_fusion,
        ]

    def compute_service_features(self, service_node):
        stats = self._stats[service_node]

        degree = len(stats["connected_peers"])

        avg_xgb = (
            float(np.mean(stats["xgb_probs"]))
            if stats["xgb_probs"] else 0.0
        )

        max_ae = (
            float(np.max(stats["ae_scores"]))
            if stats["ae_scores"] else 0.0
        )

        avg_fusion = (
            float(np.mean(stats["fusion_scores"]))
            if stats["fusion_scores"] else 0.0
        )

        return [
            float(degree),
            float(stats["bytes"]),
            float(stats["packets"]),
            float(len(stats["connected_peers"])),
            float(len(stats["protocols"])),
            avg_xgb,
            max_ae,
            avg_fusion,
        ]

    def validate_bipartite(self):
        """Verify that every graph edge connects HOST to SERVICE."""

        for u, v in self.g.edges():
            u_type = self.g.nodes[u]["node_type"]
            v_type = self.g.nodes[v]["node_type"]

            if u_type == v_type:
                raise ValueError(
                    "BIPARTITE VIOLATION: "
                    f"Edge between {u_type} nodes: {u} <-> {v}"
                )

        return True

    def to_pyg_hetero(self):
        """
        Convert the graph into PyTorch Geometric HeteroData.

        edge_flow_ids follows exactly the same ordering as:
            edge_index
            edge_attr
            y

        It is stored as separate metadata and is not included
        in the nine-dimensional edge feature matrix.
        """

        import torch
        from torch_geometric.data import HeteroData

        pyg_data = HeteroData()

        sorted_hosts = sorted(self.host_nodes)
        sorted_services = sorted(self.service_nodes)

        host_map = {
            host: index
            for index, host in enumerate(sorted_hosts)
        }

        service_map = {
            service: index
            for index, service in enumerate(sorted_services)
        }

        host_x_list = [
            self.compute_host_features(host)
            for host in sorted_hosts
        ]

        service_x_list = [
            self.compute_service_features(service)
            for service in sorted_services
        ]

        pyg_data["host"].x = (
            torch.tensor(host_x_list, dtype=torch.float32)
            if host_x_list
            else torch.empty((0, 8), dtype=torch.float32)
        )

        pyg_data["service"].x = (
            torch.tensor(service_x_list, dtype=torch.float32)
            if service_x_list
            else torch.empty((0, 8), dtype=torch.float32)
        )

        edge_list = []
        edge_attr_list = []
        edge_y_list = []
        edge_flow_ids_list = []

        for u, v, data in self.g.edges(data=True):
            u_type = self.g.nodes[u]["node_type"]
            v_type = self.g.nodes[v]["node_type"]

            host_node = u if u_type == "HOST" else v
            service_node = v if u_type == "HOST" else u

            if (
                host_node not in host_map
                or service_node not in service_map
            ):
                continue

            host_index = host_map[host_node]
            service_index = service_map[service_node]

            for flow in data.get("flows", []):
                edge_list.append([
                    host_index,
                    service_index,
                ])

                edge_attr_list.append([
                    flow["timestamp"],
                    flow["dst_port"],
                    flow["protocol"],
                    flow["byte_count"],
                    flow["pkt_count"],
                    flow["duration"],
                    flow["xgb_prob"],
                    flow["ae_score"],
                    flow["fusion_score"],
                ])

                edge_y_list.append(
                    int(flow["attack_label"])
                )

                edge_flow_ids_list.append(
                    int(flow["flow_id"])
                )

        if edge_list:
            edge_index = (
                torch.tensor(edge_list, dtype=torch.long)
                .t()
                .contiguous()
            )

            edge_attr = torch.tensor(
                edge_attr_list,
                dtype=torch.float32,
            )

            edge_y = torch.tensor(
                edge_y_list,
                dtype=torch.long,
            )

            edge_flow_ids = torch.tensor(
                edge_flow_ids_list,
                dtype=torch.long,
            )

        else:
            edge_index = torch.empty(
                (2, 0),
                dtype=torch.long,
            )

            edge_attr = torch.empty(
                (0, len(EDGE_FEATURE_NAMES)),
                dtype=torch.float32,
            )

            edge_y = torch.empty(
                (0,),
                dtype=torch.long,
            )

            edge_flow_ids = torch.empty(
                (0,),
                dtype=torch.long,
            )

        if not (
            edge_index.shape[1]
            == edge_attr.shape[0]
            == edge_y.shape[0]
            == edge_flow_ids.shape[0]
        ):
            raise ValueError(
                "Graph edge arrays are not aligned."
            )

        if edge_flow_ids.numel() != torch.unique(edge_flow_ids).numel():
            raise ValueError(
                "Duplicate flow IDs found in graph edges."
            )

        pyg_data["host", "connects_to", "service"].edge_index = (
            edge_index
        )

        pyg_data["host", "connects_to", "service"].edge_attr = (
            edge_attr
        )

        pyg_data["host", "connects_to", "service"].y = (
            edge_y
        )

        # Metadata only: never passed as a GNN input feature.
        pyg_data["host", "connects_to", "service"].flow_id = (
            edge_flow_ids
        )

        return pyg_data, host_map, service_map

    def get_graph_summary(self):
        return {
            "num_hosts": len(self.host_nodes),
            "num_services": len(self.service_nodes),
            "num_total_nodes": self.g.number_of_nodes(),
            "num_host_service_connections": self.g.number_of_edges(),
            "num_flow_edges": len(self._seen_flow_ids),
            "host_feature_dim": len(HOST_FEATURE_NAMES),
            "service_feature_dim": len(SERVICE_FEATURE_NAMES),
            "edge_feature_dim": len(EDGE_FEATURE_NAMES),
            "is_strictly_bipartite": self.validate_bipartite(),
            "direct_label_features_excluded": True,
            "end_to_end_leakage_audit": "NOT_VERIFIED",
        }
