"""CPU-only checks of the P2.2c temperature-rest research protocol."""
from scripts.p22_thermalreset import (
    MIN_REST_SECONDS, STABLE_SECONDS, COLD_TEMP_C,
    MAX_CHILD_SECONDS, MAX_SESSION_SECONDS,
    idle_enough, RUNS
)


def test_predeclared_protocol_stays_conservative():
    assert MIN_REST_SECONDS == 900
    assert STABLE_SECONDS == 60
    assert COLD_TEMP_C == 38
    assert MAX_CHILD_SECONDS <= 150
    assert MAX_SESSION_SECONDS <= 55*60
    assert RUNS == ("grassmann","transformer","local")


def test_idleness_gate_is_joint_not_temp_only():
    safe={"gpu_temp_c":38,"gpu_util_percent":10,"gpu_vram_mib":128}
    assert idle_enough(safe)
    for key,value in (
        ("gpu_temp_c",38.1),
        ("gpu_util_percent",10.1),
        ("gpu_vram_mib",129)):
        change={**safe,key:value}
        assert not idle_enough(change)
