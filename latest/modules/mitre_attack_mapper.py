"""
MITRE ATT&CK Attack Technique Mapper

Phase 2 — MITRE ATT&CK + Threat Context Integration

Evidence-Based ATT&CK Technique Mapping for CyberGraph

Determines whether observed network behaviors match known ATT&CK techniques based on:
- Behavioral evidence from flow telemetry
- BARI route analysis
- Correlation engine findings
- Attack journey progression

Provides defensible, explainable mappings with confidence scores rather than guesses.
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MITRETechnique:
    """MITRE ATT&CK technique definition with behavioral indicators."""
    technique_id: str
    technique_name: str
    tactic: str
    description: str
    behavioral_indicators: Dict[str, float]
    required_confidence_threshold: float
    evidence_weights: Dict[str, float]


class MITREAttackMapper:
    """
    Evidence-based MITRE ATT&CK technique mapper for CyberGraph.

    Provides deterministic mapping from network behavior to ATT&CK techniques
    based on observable evidence, not ML guessing.
    """

    def __init__(self):
        # Current evidence-based technique mappings
        self.techniques = {
            "T1046": MITRETechnique(
                technique_id="T1046",
                technique_name="Network Service Discovery",
                tactic="Discovery",
                description="Adversaries scan for service configurations to identify exposed services and potential attack vectors",
                behavioral_indicators={
                    "port_diversity": 0.3,  # Multiple destination ports
                    "connection_count": 0.25,  # Many connections to different services
                    "short_durations": 0.2,  # Quick connection attempts
                    "service_probing": 0.25,  # Multiple service types
                },
                required_confidence_threshold=0.7,
                evidence_weights={
                    "port_diversity": 0.3,
                    "connection_count": 0.3,
                    "short_durations": 0.2,
                    "service_probing": 0.2,
                }
            ),

            "T1110": MITRETechnique(
                technique_id="T1110",
                technique_name="Brute Force",
                tactic="Credential Access",
                description="Adversaries attempt multiple login combinations to guess passwords",
                behavioral_indicators={
                    "repeated_connections": 0.4,  # Many connections from same source
                    "auth_patterns": 0.3,  # Authentication service attempts
                    "rate_intensity": 0.3,  # High connection rates
                },
                required_confidence_threshold=0.65,
                evidence_weights={
                    "repeated_connections": 0.4,
                    "auth_patterns": 0.3,
                    "rate_intensity": 0.3,
                }
            ),

            "T1498": MITRETechnique(
                technique_id="T1498",
                technique_name="Network Denial of Service",
                tactic="Impact",
                description="Adversaries disrupt network services by overwhelming targets with traffic",
                behavioral_indicators={
                    "volume_burst": 0.4,  # Sudden traffic volume increase
                    "connection_rate": 0.35,  # High connection rates
                    "duration_intensity": 0.25,  # Sustained high traffic
                },
                required_confidence_threshold=0.75,
                evidence_weights={
                    "volume_burst": 0.4,
                    "connection_rate": 0.35,
                    "duration_intensity": 0.25,
                }
            ),

            "T1059": MITRETechnique(
                technique_id="T1059",
                technique_name="Command and Scripting Interpreter",
                tactic="Execution",
                description="Adversaries execute commands or scripts to accomplish objectives",
                behavioral_indicators={
                    "command_patterns": 0.4,  # Known command signatures
                    "script_execution": 0.35,  # Scripting protocol usage
                    "protocol_anomalies": 0.25,  # Unusual protocol usage
                },
                required_confidence_threshold=0.8,
                evidence_weights={
                    "command_patterns": 0.4,
                    "script_execution": 0.35,
                    "protocol_anomalies": 0.25,
                }
            )
        }

    def map_flow_to_techniques(self, flow_record: Dict, bari_result: Dict,
                             correlation_result: Dict, attack_journey: Dict) -> List[Dict]:
        """
        Map a single flow record to potential MITRE ATT&CK techniques based on evidence.

        Returns:
            List of technique mappings with confidence scores, sorted by confidence
        """
        technique_matches = []

        # Extract evidence from flow record
        evidence = self._extract_flow_evidence(flow_record, bari_result, correlation_result)

        # Check each technique against the evidence
        for tech_id, technique in self.techniques.items():
            confidence = self._calculate_confidence(technique, evidence)

            if confidence >= technique.required_confidence_threshold:
                mapping = self._create_mapping(
                    technique=technique,
                    evidence=evidence,
                    confidence=confidence,
                    flow_record=flow_record,
                    bari_result=bari_result,
                    correlation_result=correlation_result,
                    attack_journey=attack_journey
                )
                technique_matches.append(mapping)

        # Sort by confidence (highest first)
        technique_matches.sort(key=lambda x: x["confidence"], reverse=True)
        return technique_matches

    def _extract_flow_evidence(self, flow_record: Dict, bari_result: Dict,
                             correlation_result: Dict) -> Dict[str, float]:
        """Extract quantitative evidence from flow record and related data."""
        evidence = {}

        # Basic flow characteristics
        dst_port = flow_record.get("dst_port", 0)
        byte_count = flow_record.get("byte_count", 0)
        pkt_count = flow_record.get("pkt_count", 0)
        duration = flow_record.get("duration", 0)

        # Port diversity (from correlation context)
        evidence["port_diversity"] = self._calculate_port_diversity(flow_record, correlation_result)

        # Connection intensity metrics with duration floor to prevent extreme values
        duration_floor = max(duration, 1.0)  # Prevent division by very small durations
        evidence["connection_count"] = pkt_count / duration_floor
        evidence["volume_burst"] = byte_count / max(pkt_count, 1)
        evidence["connection_rate"] = pkt_count / duration_floor
        evidence["short_durations"] = 1.0 if duration < 1.0 else 0.0

        # Well-known benign service ports that shouldn't trigger probing evidence
        benign_service_ports = {53, 67, 68, 123, 161, 162, 500, 4500, 5353}  # DNS, DHCP, NTP, SNMP, etc.

        # Service probing (multiple ports in short time) - exclude benign services
        evidence["service_probing"] = 1.0 if (dst_port > 0 and dst_port not in benign_service_ports) else 0.0

        # BARI-related evidence
        if bari_result:
            evidence["route_novelty"] = float(bari_result.get("route_novelty", 0.0))
            evidence["route_deviation"] = float(bari_result.get("route_deviation", 0.0))
            evidence["surface_progression"] = float(bari_result.get("surface_progression", 0.0))
        else:
            evidence["route_novelty"] = 0.0
            evidence["route_deviation"] = 0.0
            evidence["surface_progression"] = 0.0

        # GNN risk evidence
        evidence["gnn_risk"] = flow_record.get("gnn_risk", 0.0)

        # Model confidence evidence (for damping low-confidence mappings)
        evidence["xgb_prob"] = flow_record.get("xgb_prob", 0.0)
        evidence["ae_score"] = flow_record.get("ae_score", 0.0)

        # Attack type evidence (from correlation)
        attack_type = correlation_result.get("attack_type", "Normal") if correlation_result else "Normal"
        evidence["auth_patterns"] = 1.0 if attack_type in ["Exploits", "Reconnaissance"] else 0.0
        evidence["protocol_anomalies"] = 1.0 if attack_type == "Fuzzer" else 0.0

        # T1059-specific evidence: detect protocol anomalies from suspicious port+protocol combos
        # C2 ports commonly use UDP (non-standard for those services)
        c2_udp_ports = {4444, 4445, 5555, 31337}
        flow_protocol = flow_record.get("protocol", 6)
        if dst_port in c2_udp_ports and flow_protocol == 17:
            evidence["protocol_anomalies"] = max(evidence.get("protocol_anomalies", 0.0), 0.8)

        # Command patterns: detected from C2 port access with non-trivial payload sizes
        if dst_port in c2_udp_ports and byte_count > 100:
            evidence["command_patterns"] = 0.8
        else:
            evidence["command_patterns"] = 0.0

        # Script execution: detected from scripting-like patterns (high pkt rate on unusual ports)
        if dst_port in (8080, 8443) and pkt_count > 10:
            evidence["script_execution"] = 0.6
        elif dst_port in c2_udp_ports:
            evidence["script_execution"] = 0.5
        else:
            evidence["script_execution"] = 0.0

        return evidence

    def _calculate_port_diversity(self, flow_record: Dict, correlation_result: Dict) -> float:
        """Estimate port diversity from correlation timeline if available."""
        # Simplified: if correlation exists and shows multiple ports, diversity is high
        if correlation_result:
            # This would need correlation timeline data for accurate calculation
            # For now, use destination port as proxy
            dst_port = flow_record.get("dst_port", 0)
            return min(1.0, dst_port / 100.0)  # Normalize to 0-1
        return 0.0

    def _calculate_confidence(self, technique: MITRETechnique, evidence: Dict[str, float]) -> float:
        """Calculate confidence score for technique mapping."""
        weighted_score = 0.0
        total_weight = 0.0

        for evidence_key, weight in technique.evidence_weights.items():
            if evidence_key in evidence:
                # Evidence strength calculation
                evidence_value = evidence[evidence_key]
                # Scale 0-1 evidence to match technique indicators
                tech_indicator = technique.behavioral_indicators.get(evidence_key, 0.0)

                if tech_indicator > 0:
                    match_score = min(1.0, evidence_value / tech_indicator)
                    weighted_score += match_score * weight
                    total_weight += weight

        if total_weight == 0:
            return 0.0

        # Base confidence from evidence match
        base_confidence = weighted_score / total_weight

        # Model-based confidence damping: if model scores indicate benign traffic,
        # discount technique confidence to avoid false positives on low-risk flows
        model_confidence = 1.0
        if "xgb_prob" in evidence:
            xgb_prob = float(evidence["xgb_prob"])
            if xgb_prob < 0.10:
                model_confidence = xgb_prob / 0.10  # Scale 0.0-1.0
            elif xgb_prob < 0.30:
                model_confidence = 0.35 + (xgb_prob - 0.10) * 1.75  # Gradual ramp 0.35-0.70

        return base_confidence * model_confidence

    def _create_mapping(self, technique: MITRETechnique, evidence: Dict[str, float],
                       confidence: float, flow_record: Dict, bari_result: Dict,
                       correlation_result: Dict, attack_journey: Dict) -> Dict:
        """Create a technique mapping result."""
        mapping = {
            "technique_id": technique.technique_id,
            "technique_name": technique.technique_name,
            "tactic": technique.tactic,
            "confidence": round(confidence, 4),
            "rationale": self._generate_rationale(technique, evidence),
            "evidence": self._collect_technique_evidence(technique, evidence),
            "source": "evidence-based",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "src_ip": flow_record.get("src_ip", "unknown"),
            "dst_ip": flow_record.get("dst_ip", "unknown"),
            "dst_port": flow_record.get("dst_port", 0),
            "risk_score": flow_record.get("risk_score", 0.0),
            "attack_type": correlation_result.get("attack_type", "Normal") if correlation_result else "Normal",
        }

        # Add BARI context if available
        if bari_result:
            mapping["bari_score"] = float(bari_result.get("bari_score", 0.0))
            mapping["route_string"] = bari_result.get("route_string", "")

        # Add attack journey context
        if attack_journey:
            mapping["journey_path"] = attack_journey.get("journey_path", [])
            mapping["attack_objective"] = attack_journey.get("attack_objective", {})

        return mapping

    def _generate_rationale(self, technique: MITRETechnique, evidence: Dict[str, float]) -> str:
        """Generate human-readable rationale for technique mapping."""
        rationale_parts = []

        # Check which evidence sources contributed
        for evidence_key, weight in technique.evidence_weights.items():
            if evidence_key in evidence and evidence[evidence_key] > 0.3:
                rationale_parts.append(evidence_key.replace('_', ' ').title())

        if not rationale_parts:
            return f"Limited evidence supports {technique.technique_name} mapping"

        evidence_summary = ", ".join(rationale_parts[:3])  # Top 3 evidence factors
        return f"Mapped to {technique.technique_name} based on {evidence_summary}"

    def _collect_technique_evidence(self, technique: MITRETechnique, evidence: Dict[str, float]) -> List[str]:
        """Collect specific evidence points supporting the technique mapping."""
        evidence_points = []

        for evidence_key, evidence_value in evidence.items():
            if evidence_value > 0.5 and evidence_key in technique.behavioral_indicators:
                evidence_points.append(f"{evidence_key.replace('_', ' ')}: {evidence_value:.2f}")

        return evidence_points[:5]  # Limit to top 5 evidence points

    def get_supported_techniques(self) -> List[str]:
        """Return list of supported technique IDs."""
        return list(self.techniques.keys())

    def get_technique_info(self, technique_id: str) -> Optional[MITRETechnique]:
        """Get detailed information about a specific technique."""
        return self.techniques.get(technique_id)

    def explain_mapping(self, mapping: Dict) -> str:
        """Provide human-readable explanation of a technique mapping."""
        technique_id = mapping["technique_id"]
        if technique_id not in self.techniques:
            return "Technique not found in supported mappings."

        technique = self.techniques[technique_id]
        confidence = mapping["confidence"]

        explanation = f"Detected {technique.technique_name} (T{technique_id}) with {confidence:.1%} confidence. "
        explanation += f"Tactic: {technique.tactic}. "

        rationale = mapping.get("rationale", "")
        if rationale:
            explanation += f"Rationale: {rationale} "

        evidence = mapping.get("evidence", [])
        if evidence:
            explanation += f"Key evidence: {', '.join(evidence[:3])}"

        return explanation