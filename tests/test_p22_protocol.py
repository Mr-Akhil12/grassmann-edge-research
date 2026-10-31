"""P2.2 protocol invariants, deliberately runnable on a clean CPU-only CI host."""
import pytest
import torch
from scripts.p22_calibrate import create_model, matched_widths, params
from scripts.p22_guard import ensure_safe


def test_matched_parameters_and_model_output():
    offsets = (1,2,4,8)
    widths, counts, mismatch = matched_widths(64,2,8,64,offsets)
    assert mismatch < 0.02
    assert set(widths) == {"grassmann","transformer","local"}
    for kind in ("grassmann","transformer","local"):
        model=create_model(kind,64,2,8,64,offsets,ff_width=widths[kind])
        assert params(model) == counts[kind]
        ids=torch.randint(0,256,(2,10))
        with torch.inference_mode():
            assert model(ids).shape == (2,10,256)


def test_guard_trips_on_power_temperature_vram_ram():
    healthy={"ac_line":1,"gpu_temp_c":40,"gpu_vram_mib":1750,
             "free_ram_gib":19.0}
    assert ensure_safe(healthy) == healthy
    for field,value in (
        ("ac_line",0),("ac_line",255),("gpu_temp_c",78),
        ("gpu_vram_mib",3000),("free_ram_gib",7.9)):
        broken=dict(healthy)
        broken[field]=value
        with pytest.raises(RuntimeError):
            ensure_safe(broken)
