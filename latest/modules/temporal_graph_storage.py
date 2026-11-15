
import os
import torch
import numpy as np

try:
    from torch_geometric.data import HeteroData, Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class TemporalGraphSnapshot:
    """Encapsulates a single temporal bipartite graph snapshot."""

    def __init__(
        self,
        window_id,
        timestamp_start,
        timestamp_end,
        host_nodes,
        service_nodes,
        host_x,
        service_x,
        edge_index_host_service,
        edge_attr,
        edge_y,
        metadata=None,
        edge_flow_ids=None,
    ):
        self.window_id = int(window_id)
        self.timestamp_start = float(timestamp_start)
        self.timestamp_end = float(timestamp_end)

        self.host_nodes = list(host_nodes)
        self.service_nodes = list(service_nodes)

        self.host_x = np.asarray(host_x, dtype=np.float32)
        self.service_x = np.asarray(service_x, dtype=np.float32)
        self.edge_index_host_service = np.asarray(
            edge_index_host_service, dtype=np.int64
        )
        self.edge_attr = np.asarray(edge_attr, dtype=np.float32)
        self.edge_y = np.asarray(edge_y, dtype=np.int64)

        if edge_flow_ids is None:
            self.edge_flow_ids = np.full(
                len(self.edge_y), -1, dtype=np.int64
            )
        else:
            self.edge_flow_ids = np.asarray(
                edge_flow_ids, dtype=np.int64
            )

        if len(self.edge_flow_ids) != len(self.edge_y):
            raise ValueError(
                "Flow IDs and edge labels do not align."
            )

        if self.edge_attr.shape[0] != len(self.edge_y):
            raise ValueError(
                "Edge attributes and labels do not align."
            )

        self.metadata = metadata or {}

    def to_pyg_hetero(self):
        """Convert snapshot into PyTorch Geometric HeteroData."""

        data = HeteroData()

        data.window_id = self.window_id
        data.timestamp_start = self.timestamp_start
        data.timestamp_end = self.timestamp_end

        data["host"].x = torch.tensor(
            self.host_x, dtype=torch.float32
        )
        data["service"].x = torch.tensor(
            self.service_x, dtype=torch.float32
        )

        data["host"].node_ids = self.host_nodes
        data["service"].node_ids = self.service_nodes

        edge_type = ("host", "connects_to", "service")

        if self.edge_index_host_service.size > 0:
            data[edge_type].edge_index = torch.tensor(
                self.edge_index_host_service,
                dtype=torch.long,
            )

            data[edge_type].edge_attr = torch.tensor(
                self.edge_attr,
                dtype=torch.float32,
            )

            data[edge_type].y = torch.tensor(
                self.edge_y,
                dtype=torch.long,
            )

            data[edge_type].flow_id = torch.tensor(
                self.edge_flow_ids,
                dtype=torch.long,
            )

        else:
            feature_dim = (
                self.edge_attr.shape[1]
                if self.edge_attr.ndim > 1
                else 0
            )

            data[edge_type].edge_index = torch.empty(
                (2, 0), dtype=torch.long
            )
            data[edge_type].edge_attr = torch.empty(
                (0, feature_dim), dtype=torch.float32
            )
            data[edge_type].y = torch.empty(
                (0,), dtype=torch.long
            )
            data[edge_type].flow_id = torch.empty(
                (0,), dtype=torch.long
            )

        return data

    def to_dict(self):
        return {
            "window_id": self.window_id,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "host_nodes": self.host_nodes,
            "service_nodes": self.service_nodes,
            "host_x": self.host_x.tolist(),
            "service_x": self.service_x.tolist(),
            "edge_index_host_service":
                self.edge_index_host_service.tolist(),
            "edge_attr": self.edge_attr.tolist(),
            "edge_y": self.edge_y.tolist(),
            "edge_flow_ids": self.edge_flow_ids.tolist(),
            "metadata": self.metadata,
        }


class TemporalGraphStorage:
    def __init__(self, snapshots=None):
        self.snapshots = snapshots or []

    def add_snapshot(self, snapshot: TemporalGraphSnapshot):
        self.snapshots.append(snapshot)

    def save(self, filepath):
        os.makedirs(
            os.path.dirname(os.path.abspath(filepath)),
            exist_ok=True,
        )

        dataset_payload = {
            "num_snapshots": len(self.snapshots),
            "snapshots": [
                snapshot.to_dict()
                for snapshot in self.snapshots
            ],
        }

        torch.save(dataset_payload, filepath)

        print(
            f"Saved {len(self.snapshots)} temporal graph "
            f"snapshots to: {filepath}"
        )

    @classmethod
    def load(cls, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Temporal graph dataset not found: {filepath}"
            )

        payload = torch.load(
            filepath,
            map_location="cpu",
            weights_only=False,
        )

        storage = cls()

        for snapshot_dict in payload["snapshots"]:
            snapshot = TemporalGraphSnapshot(
                window_id=snapshot_dict["window_id"],
                timestamp_start=snapshot_dict["timestamp_start"],
                timestamp_end=snapshot_dict["timestamp_end"],
                host_nodes=snapshot_dict["host_nodes"],
                service_nodes=snapshot_dict["service_nodes"],
                host_x=snapshot_dict["host_x"],
                service_x=snapshot_dict["service_x"],
                edge_index_host_service=
                    snapshot_dict["edge_index_host_service"],
                edge_attr=snapshot_dict["edge_attr"],
                edge_y=snapshot_dict["edge_y"],
                edge_flow_ids=snapshot_dict.get("edge_flow_ids"),
                metadata=snapshot_dict.get("metadata"),
            )

            storage.add_snapshot(snapshot)

        print(
            f"Loaded {len(storage.snapshots)} temporal graph "
            f"snapshots from: {filepath}"
        )

        return storage

    def get_pyg_snapshot(self, window_idx):
        if window_idx < 0 or window_idx >= len(self.snapshots):
            raise IndexError(
                f"Snapshot index {window_idx} out of range."
            )

        return self.snapshots[window_idx].to_pyg_hetero()

    def get_previous_current_next(self, window_idx):
        curr_g = self.get_pyg_snapshot(window_idx)

        prev_g = (
            self.get_pyg_snapshot(window_idx - 1)
            if window_idx > 0 else None
        )

        next_g = (
            self.get_pyg_snapshot(window_idx + 1)
            if window_idx < len(self.snapshots) - 1 else None
        )

        return prev_g, curr_g, next_g
