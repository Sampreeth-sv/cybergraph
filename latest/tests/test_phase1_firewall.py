import pytest

from modules.firewall_response_executor import (
    FirewallResponseExecutor,
)


@pytest.fixture
def firewall():
    return FirewallResponseExecutor(enable_real_blocking=False)


def test_simulated_block(firewall):
    result = firewall.execute_block(
        "192.168.1.100"
    )

    assert result["status"] == "SIMULATED_BLOCK_RECORDED"
    assert result["verification"]["is_blocked"] is True


def test_simulated_unblock(firewall):
    firewall.execute_block("192.168.1.100")

    result = firewall.execute_unblock(
        "192.168.1.100"
    )

    assert result["status"] == "UNBLOCKED_SUCCESS"
    assert result["verification"]["is_blocked"] is False


def test_invalid_ip_rejected(firewall):
    with pytest.raises(ValueError):
        firewall.execute_block("invalid-ip")


def test_invalid_ip_unblock_rejected(firewall):
    with pytest.raises(ValueError):
        firewall.execute_unblock("invalid-ip")


def test_multiple_simulated_ips(firewall):
    firewall.execute_block("192.168.1.100")
    firewall.execute_block("192.168.1.101")

    assert firewall.verify_blocked(
        "192.168.1.100"
    )["is_blocked"] is True

    assert firewall.verify_blocked(
        "192.168.1.101"
    )["is_blocked"] is True

    firewall.execute_unblock("192.168.1.100")

    assert firewall.verify_blocked(
        "192.168.1.100"
    )["is_blocked"] is False

    assert firewall.verify_blocked(
        "192.168.1.101"
    )["is_blocked"] is True