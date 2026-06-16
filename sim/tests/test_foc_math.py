# SPDX-License-Identifier: MIT
"""FOC stages 2-3: bit-for-bit parity of the RTL math primitives against the
Python reference (sim/scripts/foc_reference.py).

The combinational harness rtl/foc_math.v (built as the `focmath` pybind
module) exposes sincos/clarke/park/inv_park/svpwm. Every assertion is exact
integer equality - the RTL and the reference must agree to the bit, which is
what licenses using the reference as the executable spec in later stages.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import foc_reference as fr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = PROJECT_ROOT / "sim" / "build" / "cpp"


def s18(x):
    """Wrap to signed 18-bit, matching the RTL's [17:0] truncation."""
    x &= 0x3FFFF
    return x - 0x40000 if (x & 0x20000) else x


@pytest.fixture(scope="module")
def fm(bldcsim):  # bldcsim builds the module set and adds BUILD_DIR to path
    if str(BUILD_DIR) not in sys.path:
        sys.path.insert(0, str(BUILD_DIR))
    import focmath
    return focmath.FocMath()


@pytest.fixture(scope="module")
def bits(params):
    return int(params.value("foc.sincos_table_bits"))


@pytest.fixture(scope="module")
def half(params):
    clk = params.value("rtl.clock_frequency")
    return int(round(clk / (2.0 * params.value("pwm.frequency"))))


# Deterministic input vectors spanning sign quadrants and magnitudes.
THETAS = list(range(0, 65536, 1031)) + [0, 16384, 32768, 49152, 65535]
IAB = [(0, 0), (1000, -500), (-1500, 700), (2000, 2000), (-2047, -2047),
       (300, -1200), (1234, 567), (-800, 1900)]
VDQ = [(0, 0), (200, 0), (0, 300), (-150, 250), (400, -400), (594, 0),
       (-300, -300), (123, 456)]
VAB = [(0, 0), (300, 0), (0, 400), (-250, 300), (500, -500), (594, 100),
       (-594, -100), (312, 312)]


def _ref_sincos(theta, bits):
    return fr.sincos_fx(theta, bits)


@pytest.mark.parametrize("theta", THETAS)
def test_sincos_parity(fm, bits, theta):
    out = fm.eval(theta, 0, 0, 0, 0, 0, 0)
    cos_ref, sin_ref = _ref_sincos(theta, bits)
    assert out["sin"] == sin_ref, f"sin @ {theta}"
    assert out["cos"] == cos_ref, f"cos @ {theta}"


def test_sincos_is_unit_circle(fm, bits):
    """sin^2 + cos^2 ~ 1 (Q15) - a sanity net on the table itself."""
    for theta in THETAS:
        out = fm.eval(theta, 0, 0, 0, 0, 0, 0)
        mag2 = out["sin"] ** 2 + out["cos"] ** 2
        # Q15 unit = 32767^2; allow LUT quantization slack.
        assert abs(mag2 - 32767 ** 2) < 32767 * 60, f"|sincos| @ {theta}"


@pytest.mark.parametrize("ia,ib", IAB)
def test_clarke_parity(fm, ia, ib):
    out = fm.eval(0, ia, ib, 0, 0, 0, 0)
    ialpha_ref, ibeta_ref = fr.clarke_fx(ia, ib)
    assert out["ialpha"] == s18(ialpha_ref)
    assert out["ibeta"] == s18(ibeta_ref)


@pytest.mark.parametrize("ia,ib", IAB)
@pytest.mark.parametrize("theta", [0, 8000, 16384, 30000, 49152, 60000])
def test_park_parity(fm, bits, ia, ib, theta):
    out = fm.eval(theta, ia, ib, 0, 0, 0, 0)
    cos_ref, sin_ref = _ref_sincos(theta, bits)
    ialpha_ref, ibeta_ref = fr.clarke_fx(ia, ib)
    id_ref, iq_ref = fr.park_fx(s18(ialpha_ref), s18(ibeta_ref),
                                cos_ref, sin_ref)
    assert out["id"] == s18(id_ref), f"id @ theta={theta} ia={ia} ib={ib}"
    assert out["iq"] == s18(iq_ref), f"iq @ theta={theta} ia={ia} ib={ib}"


@pytest.mark.parametrize("vd,vq", VDQ)
@pytest.mark.parametrize("theta", [0, 8000, 16384, 30000, 49152, 60000])
def test_inv_park_parity(fm, bits, vd, vq, theta):
    out = fm.eval(theta, 0, 0, vd, vq, 0, 0)
    cos_ref, sin_ref = _ref_sincos(theta, bits)
    va_ref, vb_ref = fr.inv_park_fx(vd, vq, cos_ref, sin_ref)
    assert out["valpha"] == s18(va_ref), f"valpha @ theta={theta}"
    assert out["vbeta"] == s18(vb_ref), f"vbeta @ theta={theta}"


@pytest.mark.parametrize("valpha,vbeta", VAB)
def test_svpwm_parity(fm, half, valpha, vbeta):
    out = fm.eval(0, 0, 0, 0, 0, valpha, vbeta)
    da, db, dc = fr.svpwm_fx(valpha, vbeta, half)
    assert (out["duty_a"], out["duty_b"], out["duty_c"]) == (da, db, dc), (
        f"svpwm @ ({valpha},{vbeta})")


def test_svpwm_duties_in_range(fm, half):
    """Every leg duty stays in [0, HALF] across the input set (the injection's
    job - no leg ever saturates outside the modulator's range)."""
    for valpha, vbeta in VAB:
        out = fm.eval(0, 0, 0, 0, 0, valpha, vbeta)
        for k in ("duty_a", "duty_b", "duty_c"):
            assert 0 <= out[k] <= half, f"{k}={out[k]} out of [0,{half}]"


def test_svpwm_bus_utilization(fm, half):
    """min/max injection extends linear range ~15% past sine PWM: a phase
    voltage request that pure-sine PWM would clip (|valpha| up to ~1.15*HALF/2)
    stays in range here. Check a request beyond the sine limit is unclipped."""
    # Pure-sine would clip a phase peak above HALF/2 (= center). With
    # injection, a vector of magnitude up to ~HALF/sqrt(3)*... stays linear.
    # Use a pure-alpha request just past the sine limit (center) and confirm
    # the modulator does not rail.
    over = int(0.57 * half)  # ~ inscribed-circle radius, > center (0.5*half)
    out = fm.eval(0, 0, 0, 0, 0, over, 0)
    duties = [out["duty_a"], out["duty_b"], out["duty_c"]]
    assert max(duties) < half and min(duties) > 0, (
        f"injection failed to keep {over} (> center {half // 2}) linear: "
        f"{duties}")
