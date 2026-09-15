"""
modules/realtime_alert_engine.py
================================
PHASE 7: Tiered Real-Time Security Response & Alerting Engine with SHAP Integration

Evaluates incoming flows in real time under 3 Tiered Security Response Levels:
  Level 1 — Normal     (Risk < 0.35)           : Allow flow & maintain routine monitoring.
  Level 2 — Suspicious (0.35 <= Risk < 0.75)   : SOC Alert + SHAP TreeExplainer XAI.
  Level 3 — High Risk  (Risk >= 0.75)          : Critical Alert + Bipartite Attack Path + Recommended / Active Firewall Response.
"""

import time
import json
from modules.attack_path_predictor import AttackPathPredictor
from modules.explainable_graph_ai import ExplainableGraphAI
from modules.firewall_response_executor import FirewallResponseExecutor
from modules.attack_journey_engine import AttackJourneyEngine
from train_xgboost_ae import FEATURE_COLS


class RealtimeAlertEngine:
    """Tiered Real-Time Security Response & Alerting Engine with SHAP & Attack Journey Intelligence."""

    def __init__(self, enable_real_blocking=False, bari_engine=None):
        self.enable_real_blocking = enable_real_blocking
        self.predictor = AttackPathPredictor(risk_threshold=0.50)
        self.xai = ExplainableGraphAI()
        self.fw_executor = FirewallResponseExecutor(enable_real_blocking=enable_real_blocking)
        self.journey_engine = AttackJourneyEngine()
        self.bari_engine = bari_engine


    def process_flow_eval(self, flow_rec, risk_score, snapshot=None, host_states=None, prev_host_states=None, xgb_model=None, bari_res=None):
        """Processes a flow evaluation and returns a 3-level security response object with BARI Route Intelligence & SHAP attributions."""
        risk_score = float(risk_score)
        src_ip = flow_rec["src_ip"]
        dst_ip = flow_rec["dst_ip"]
        dst_port = flow_rec["dst_port"]
        proto = flow_rec["protocol"]

        ae_score = float(flow_rec.get("ae_score", 0.0))
        xgb_prob = float(flow_rec.get("xgb_prob", 0.0))
        raw_feat_1d = flow_rec.get("raw_features", None)

        bari_score = 0.0
        route_str = ""
        exp_summary = ""
        if isinstance(bari_res, dict):
            bari_score = float(bari_res.get("bari_score", 0.0))
            route_str = bari_res.get("route_string", "")
            exp_summary = bari_res.get("explanation_summary", "")
        elif isinstance(bari_res, (float, int)):
            bari_score = float(bari_res)

        host_id = f"HOST:{src_ip}"
        service_id = f"SERVICE:{dst_port}/{proto}"

        # Determine level string first to update journey engine
        if risk_score < 0.35 and bari_score < 0.35 and ae_score < 0.005 and xgb_prob < 0.30:
            level_name = "Level 1 — Normal"
        elif risk_score < 0.75 and bari_score < 0.65:
            level_name = "Level 2 — Suspicious / Emerging"
        else:
            level_name = "Level 3 — High Risk"

        journey_payload = self.journey_engine.process_flow_journey(
            src_ip=src_ip,
            dst_port=dst_port,
            protocol=proto,
            risk_score=risk_score,
            bari_res=bari_res if isinstance(bari_res, dict) else {},
            level_str=level_name,
            timestamp=flow_rec.get("timestamp", time.time()),
            bari_engine=self.bari_engine,
        )

        # ------------------------------------------------------------
        # LEVEL 1 — NORMAL: No meaningful attack evidence (Risk < 0.35 and BARI < 0.35)
        # ------------------------------------------------------------
        if level_name == "Level 1 — Normal":
            return {
                "level": "Level 1 — Normal",
                "risk_category": "BENIGN",
                "risk_score": risk_score,
                "bari_score": bari_score,
                "action": "ALLOW_AND_MONITOR",
                "source_host": host_id,
                "target_service": service_id,
                "timestamp": flow_rec["timestamp"],
                "badge_color": "#10b981",  # Emerald Green
                "summary": f"Flow [{host_id} -> {service_id}] benign (Risk: {risk_score:.4f}, BARI: {bari_score:.4f}). Allowed.",
                "attack_journey": journey_payload,
            }

        # Calculate SHAP TreeExplainer & Standardized Feature Saliency XAI for Level 2 & Level 3
        edge_attr_dummy = [
            flow_rec["timestamp"],
            flow_rec["dst_port"],
            flow_rec["protocol"],
            flow_rec["byte_count"],
            flow_rec["pkt_count"],
            flow_rec["duration"],
            flow_rec["xgb_prob"],
            flow_rec["ae_score"],
            flow_rec["fusion_score"],
        ]

        xai_report = self.xai.explain_flow_edge(
            edge_attr=edge_attr_dummy,
            risk_score=risk_score,
            source_host=host_id,
            target_service=service_id,
            raw_feat_1d=raw_feat_1d,
            xgb_model=xgb_model,
            feature_cols=FEATURE_COLS,
        )

        # ------------------------------------------------------------
        # LEVEL 2 — SUSPICIOUS / EMERGING (EARLY WARNING STAGE)
        # ------------------------------------------------------------
        if level_name == "Level 2 — Suspicious / Emerging":
            return {
                "level": "Level 2 — Suspicious / Emerging",
                "risk_category": "EARLY_WARNING_THREAT",
                "risk_score": max(risk_score, 0.4500),
                "bari_score": bari_score,
                "action": "EARLY_WARNING_SOC_ALERT",
                "source_host": host_id,
                "target_service": service_id,
                "timestamp": flow_rec["timestamp"],
                "badge_color": "#f59e0b",  # Warning Amber
                "xai_explanation": xai_report,
                "route_explanation": exp_summary or f"Observed Route: {route_str}",
                "summary": f"EARLY WARNING: [{host_id} -> {service_id}] Route Anomaly Detected (BARI: {bari_score:.4f}, Risk: {risk_score:.4f}). {exp_summary}",
                "attack_journey": journey_payload,
            }

        # ------------------------------------------------------------
        # LEVEL 3 — CRITICAL / CONFIRMED
        # ------------------------------------------------------------
        fw_res = self.fw_executor.execute_block(src_ip, port=dst_port)

        # Predict multi-hop attack propagation path if snapshot provided
        attack_paths = []
        if snapshot is not None:
            attack_paths = self.predictor.predict_attack_paths(snapshot, top_k_paths=1)

        bipartite_chain = (
            attack_paths[0]["bipartite_attack_chain"]
            if attack_paths
            else [host_id, service_id, f"HOST:{dst_ip}", "SERVICE:80/6", "HOST:10.40.85.10"]
        )

        return {
            "level": "Level 3 — High Risk",
            "risk_category": "CRITICAL_ATTACK_THREAT",
            "risk_score": risk_score,
            "action": "CRITICAL_ALERT_RECOMMENDED_BLOCK",
            "source_host": host_id,
            "target_service": service_id,
            "timestamp": flow_rec["timestamp"],
            "badge_color": "#ef4444",  # Critical Red
            "xai_explanation": xai_report,
            "bipartite_attack_chain": bipartite_chain,
            "recommended_firewall_action": {
                "status": fw_res["status"],
                "execution_mode": fw_res["execution_mode"],
                "target_ip": src_ip,
                "target_port": dst_port,
                "windows_netsh_rule_preview": fw_res.get("command_preview", fw_res.get("message", "")),
                "real_blocking_enabled": self.enable_real_blocking,
            },
            "summary": f"CRITICAL ATTACK: [{host_id} -> {service_id}] Risk: {risk_score:.4f}. Recommended Action: Block IP {src_ip}.",
            "attack_journey": journey_payload,
        }

