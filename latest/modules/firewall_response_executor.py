
"""
modules/firewall_response_executor.py

OS FIREWALL & ACL RESPONSE EXECUTION MANAGER

Supports:
1. Simulated in-memory blocking.
2. Real Windows Defender Firewall blocking.
3. Real Linux iptables blocking.
4. Block verification and unblocking.

Real firewall execution requires appropriate OS privileges.
"""

import ipaddress
import logging
import platform
import subprocess

logger = logging.getLogger(__name__)


class FirewallResponseExecutor:

    def __init__(self, enable_real_blocking=False):
        self.enable_real_blocking = enable_real_blocking
        self.os_type = platform.system().lower()

        self.simulated_acl_blocked_ips = set()
        self.active_os_firewall_rules = {}

    @staticmethod
    def _validate_ip(ip_address):
        """Accept only valid IPv4 addresses supported by this implementation."""
        try:
            return str(ipaddress.IPv4Address(str(ip_address).strip()))
        except ipaddress.AddressValueError as exc:
            raise ValueError("A valid IPv4 address is required.") from exc

    def _commands(self, ip_address):
        """Return argument lists for safe shell-free execution."""

        ip_address = self._validate_ip(ip_address)

        if self.os_type == "windows":
            rule_name = f"Block-CyberGraph-{ip_address}"

            block = [
                "netsh", "advfirewall", "firewall", "add", "rule",
                f"name={rule_name}",
                "dir=in",
                "action=block",
                f"remoteip={ip_address}",
            ]

            unblock = [
                "netsh", "advfirewall", "firewall", "delete", "rule",
                f"name={rule_name}",
            ]

            verify = [
                "netsh", "advfirewall", "firewall", "show", "rule",
                f"name={rule_name}",
            ]

        elif self.os_type == "linux":
            block = [
                "iptables", "-A", "INPUT",
                "-s", ip_address, "-j", "DROP",
            ]

            unblock = [
                "iptables", "-D", "INPUT",
                "-s", ip_address, "-j", "DROP",
            ]

            verify = [
                "iptables", "-C", "INPUT",
                "-s", ip_address, "-j", "DROP",
            ]

        else:
            raise RuntimeError(
                f"Unsupported firewall platform: {self.os_type}"
            )

        return block, unblock, verify

    def generate_rule_preview(self, ip_address, port=None):
        ip_address = self._validate_ip(ip_address)
        block, unblock, verify = self._commands(ip_address)

        return {
            "os_platform": self.os_type,
            "ip_address": ip_address,
            "port": port,
            "block_command_preview": " ".join(block),
            "unblock_command_preview": " ".join(unblock),
            "verify_command_preview": " ".join(verify),
        }

    def verify_blocked(self, ip_address):
        ip_address = self._validate_ip(ip_address)

        is_sim_blocked = (
            ip_address in self.simulated_acl_blocked_ips
        )

        is_real_blocked = False

        if ip_address in self.active_os_firewall_rules:
            try:
                _, _, verify_cmd = self._commands(ip_address)

                result = subprocess.run(
                    verify_cmd,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    shell=False,
                )

                is_real_blocked = result.returncode == 0

            except (OSError, subprocess.SubprocessError) as exc:
                logger.error(
                    "Firewall verification failed for %s: %s",
                    ip_address,
                    exc,
                )

        return {
            "ip_address": ip_address,
            "is_blocked": is_sim_blocked or is_real_blocked,
            "simulated_memory_blocked": is_sim_blocked,
            "real_os_firewall_blocked": is_real_blocked,
            "status_description": (
                "BLOCKED & VERIFIED"
                if is_sim_blocked or is_real_blocked
                else "UNBLOCKED / ALLOWED"
            ),
        }

    def execute_block(self, ip_address, port=None, force_real=False):
        ip_address = self._validate_ip(ip_address)

        preview = self.generate_rule_preview(ip_address, port)

        should_execute_real = (
            force_real or self.enable_real_blocking
        )

        # SIMULATED MODE
        if not should_execute_real:
            self.simulated_acl_blocked_ips.add(ip_address)

            return {
                "status": "SIMULATED_BLOCK_RECORDED",
                "execution_mode": "SIMULATED_MEMORY_ONLY",
                "ip_address": ip_address,
                "message": (
                    f"IP {ip_address} added to the simulated block list."
                ),
                "command_preview": preview["block_command_preview"],
                "verification": self.verify_blocked(ip_address),
            }

        # REAL FIREWALL MODE
        try:
            block_cmd, unblock_cmd, _ = self._commands(ip_address)

            result = subprocess.run(
                block_cmd,
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
            )

            if result.returncode != 0:
                return {
                    "status": "FIREWALL_EXECUTION_FAILED",
                    "execution_mode": "REAL_OS_FIREWALL",
                    "ip_address": ip_address,
                    "message": "The operating system rejected the firewall rule.",
                    "error": result.stderr.strip() or result.stdout.strip(),
                    "verification": self.verify_blocked(ip_address),
                }

            # Store the rule only after the OS accepts it.
            self.active_os_firewall_rules[ip_address] = unblock_cmd

            verification = self.verify_blocked(ip_address)

            if not verification["real_os_firewall_blocked"]:
                return {
                    "status": "FIREWALL_VERIFICATION_FAILED",
                    "execution_mode": "REAL_OS_FIREWALL",
                    "ip_address": ip_address,
                    "message": (
                        "The command completed, but the firewall rule "
                        "could not be verified."
                    ),
                    "verification": verification,
                }

            return {
                "status": "REAL_FIREWALL_RULE_APPLIED",
                "execution_mode": "REAL_OS_FIREWALL",
                "ip_address": ip_address,
                "message": f"Firewall rule verified for {ip_address}.",
                "cmd_output": result.stdout.strip(),
                "verification": verification,
            }

        except (OSError, subprocess.SubprocessError) as exc:
            logger.exception("Firewall block failed for %s", ip_address)

            return {
                "status": "FIREWALL_EXECUTION_FAILED",
                "execution_mode": "REAL_OS_FIREWALL",
                "ip_address": ip_address,
                "message": "Firewall command execution failed.",
                "error": str(exc),
                "verification": self.verify_blocked(ip_address),
            }

    def execute_unblock(self, ip_address):
        ip_address = self._validate_ip(ip_address)

        # Remove a simulated block.
        if ip_address in self.simulated_acl_blocked_ips:
            self.simulated_acl_blocked_ips.remove(ip_address)

        # Remove a real OS firewall rule if one is tracked.
        if ip_address in self.active_os_firewall_rules:
            unblock_cmd = self.active_os_firewall_rules[ip_address]

            try:
                result = subprocess.run(
                    unblock_cmd,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    shell=False,
                )

                if result.returncode != 0:
                    return {
                        "status": "UNBLOCK_FAILED",
                        "ip_address": ip_address,
                        "message": "The operating system could not remove the firewall rule.",
                        "error": result.stderr.strip() or result.stdout.strip(),
                        "verification": self.verify_blocked(ip_address),
                    }

                # Remove the tracked rule only after successful execution.
                self.active_os_firewall_rules.pop(ip_address, None)

            except (OSError, subprocess.SubprocessError) as exc:
                logger.exception(
                    "Firewall unblock failed for %s", ip_address
                )

                return {
                    "status": "UNBLOCK_FAILED",
                    "ip_address": ip_address,
                    "message": "Firewall unblock command failed.",
                    "error": str(exc),
                    "verification": self.verify_blocked(ip_address),
                }

        verification = self.verify_blocked(ip_address)

        if verification["is_blocked"]:
            return {
                "status": "UNBLOCK_VERIFICATION_FAILED",
                "ip_address": ip_address,
                "message": "The IP is still reported as blocked.",
                "verification": verification,
            }

        return {
            "status": "UNBLOCKED_SUCCESS",
            "ip_address": ip_address,
            "message": f"IP {ip_address} successfully unblocked.",
            "verification": verification,
        }
