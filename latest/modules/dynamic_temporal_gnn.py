
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroBipartiteConv(nn.Module):
    """Bipartite graph convolution: Host <-> Service."""

    def __init__(self, host_dim, service_dim, hidden_dim):
        super().__init__()
        self.host_to_service = nn.Linear(host_dim, hidden_dim)
        self.service_to_host = nn.Linear(service_dim, hidden_dim)
        self.host_self = nn.Linear(host_dim, hidden_dim)
        self.service_self = nn.Linear(service_dim, hidden_dim)

    def forward(self, host_x, service_x, edge_index):
        num_hosts = host_x.shape[0]
        num_services = service_x.shape[0]
        hidden_dim = self.host_self.out_features

        msg_to_service = torch.zeros(
            (num_services, hidden_dim), device=host_x.device
        )
        msg_to_host = torch.zeros(
            (num_hosts, hidden_dim), device=host_x.device
        )

        if edge_index.shape[1] > 0:
            h_idx, s_idx = edge_index[0], edge_index[1]

            h_trans = F.relu(self.host_to_service(host_x))
            s_trans = F.relu(self.service_to_host(service_x))

            msg_to_service.index_add_(0, s_idx, h_trans[h_idx])
            msg_to_host.index_add_(0, h_idx, s_trans[s_idx])

            s_deg = torch.zeros(
                (num_services, 1), device=host_x.device
            ).index_add_(
                0, s_idx,
                torch.ones((len(s_idx), 1), device=host_x.device)
            ).clamp(min=1.0)

            h_deg = torch.zeros(
                (num_hosts, 1), device=host_x.device
            ).index_add_(
                0, h_idx,
                torch.ones((len(h_idx), 1), device=host_x.device)
            ).clamp(min=1.0)

            msg_to_service = msg_to_service / s_deg
            msg_to_host = msg_to_host / h_deg

        out_host = F.relu(self.host_self(host_x) + msg_to_host)
        out_service = F.relu(self.service_self(service_x) + msg_to_service)

        return out_host, out_service


class TemporalNodeUpdate(nn.Module):
    """GRU temporal updater with node-ID-based state alignment."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.host_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.service_gru = nn.GRUCell(hidden_dim, hidden_dim)

    @staticmethod
    def align_previous_state(current_ids, previous_ids, previous_state, current_state):
        """
        Align previous hidden states to current node order using node IDs.
        New nodes receive zero-initialized hidden states.
        """
        if previous_state is None or previous_ids is None:
            return torch.zeros_like(current_state)

        if len(previous_ids) != previous_state.shape[0]:
            raise ValueError("Previous node IDs and hidden states have different lengths.")

        previous_index = {
            node_id: index for index, node_id in enumerate(previous_ids)
        }

        zero_state = torch.zeros(
            current_state.shape[1],
            dtype=current_state.dtype,
            device=current_state.device,
        )

        aligned = [
            previous_state[previous_index[node_id]]
            if node_id in previous_index
            else zero_state
            for node_id in current_ids
        ]

        if not aligned:
            return current_state.new_zeros(current_state.shape)

        return torch.stack(aligned, dim=0)

    def forward(
        self,
        curr_host,
        curr_service,
        prev_host_h=None,
        prev_service_h=None,
        host_ids=None,
        service_ids=None,
        prev_host_ids=None,
        prev_service_ids=None,
    ):
        # Require IDs whenever previous states are supplied.
        if prev_host_h is not None and (
            host_ids is None or prev_host_ids is None
        ):
            raise ValueError("Host IDs are required for temporal state alignment.")

        if prev_service_h is not None and (
            service_ids is None or prev_service_ids is None
        ):
            raise ValueError("Service IDs are required for temporal state alignment.")

        prev_host_h = self.align_previous_state(
            host_ids, prev_host_ids, prev_host_h, curr_host
        )
        prev_service_h = self.align_previous_state(
            service_ids, prev_service_ids, prev_service_h, curr_service
        )

        next_host_h = self.host_gru(curr_host, prev_host_h)
        next_service_h = self.service_gru(curr_service, prev_service_h)

        return next_host_h, next_service_h


class EdgeRiskClassifier(nn.Module):
    """Edge classification head."""

    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        in_dim = hidden_dim * 2 + edge_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, host_states, service_states, edge_index, edge_attr):
        if edge_index.shape[1] == 0:
            return torch.empty(
                (0,), device=host_states.device
            )

        h_idx, s_idx = edge_index[0], edge_index[1]
        h_emb = host_states[h_idx]
        s_emb = service_states[s_idx]

        cat_feat = torch.cat([h_emb, s_emb, edge_attr], dim=1)
        return self.mlp(cat_feat).squeeze(-1)


class DynamicBipartiteTemporalGNN(nn.Module):
    """Dynamic Bipartite Temporal GNN."""

    def __init__(self, host_dim=8, service_dim=8, edge_dim=9, hidden_dim=32):
        super().__init__()
        self.conv1 = HeteroBipartiteConv(
            host_dim, service_dim, hidden_dim
        )
        self.temporal_update = TemporalNodeUpdate(hidden_dim)
        self.classifier = EdgeRiskClassifier(hidden_dim, edge_dim)

    def forward(
        self,
        host_x,
        service_x,
        edge_index,
        edge_attr,
        prev_host_h=None,
        prev_service_h=None,
        host_ids=None,
        service_ids=None,
        prev_host_ids=None,
        prev_service_ids=None,
    ):
        if host_ids is None or service_ids is None:
            raise ValueError("Current host_ids and service_ids must be provided.")

        if len(host_ids) != host_x.shape[0]:
            raise ValueError("host_ids do not match host feature rows.")

        if len(service_ids) != service_x.shape[0]:
            raise ValueError("service_ids do not match service feature rows.")

        h_conv, s_conv = self.conv1(
            host_x, service_x, edge_index
        )

        h_state, s_state = self.temporal_update(
            h_conv,
            s_conv,
            prev_host_h,
            prev_service_h,
            host_ids,
            service_ids,
            prev_host_ids,
            prev_service_ids,
        )

        logits = self.classifier(
            h_state, s_state, edge_index, edge_attr
        )

        return logits, h_state, s_state
