# FOC Fixed-Point Convention

The single source of truth for the Q-format and saturation rules used by the
FOC RTL (`rtl/clarke.v`, `park.v`, `inv_park.v`, `sincos.v`, `current_pi.v`,
`svpwm.v`) and mirrored exactly by the numpy reference
(`sim/scripts/foc_reference.py`). The parity tests
(`sim/tests/test_foc_math.py`) enforce that the two agree bit-for-bit-close.
Referenced by [foc-checklist](foc-checklist.md). Keep this in sync with both
the RTL and the reference; a divergence here is a bug in one of the three.

## Units and word widths

| Quantity        | Format                | Range (nominal)        | Notes |
|-----------------|-----------------------|------------------------|-------|
| Phase current   | signed, **ADC LSB**   | ±2048                  | code − offset; 1 LSB = vref/4096 / (amp_gain·shunt) amps. The loop never converts to amps — it controls in LSB. |
| iα, iβ          | signed LSB            | ±~2400                 | Clarke can exceed a single phase by 2/√3. |
| id, iq          | signed LSB            | ±~2400                 | Park is a rotation: |i_dq| = |i_αβ|. |
| Electrical angle| unsigned **Q16**      | 0..65535 = 0..2π       | 12-bit sensor × pole_pairs, wrapped to 16 bits. |
| sin / cos       | signed **Q15**        | −32768..32767 = −1..+1 | quarter-wave LUT, `sincos_table_bits` address bits. |
| vd, vq          | signed **duty units** | ±HALF                  | HALF = `PWM_HALF_PERIOD`. A magnitude of HALF = full half-bus phase drive. |
| vα, vβ          | signed duty units     | ±HALF                  | inverse Park of vd,vq. |
| duty[k]         | unsigned              | 0..HALF                | per-leg `duty_compare` into `pwm_generator`. |

All intermediate products are computed in **signed 32-bit** and shifted back;
every result that lands in a state register or a module output is
**saturated** to its declared width (no wraparound). Saturation helpers:
`sat16(x)` clamps to ±32767, `sat_duty(x)` clamps to [0, HALF].

## Constants (Q15)

- `ONE_OVER_SQRT3 = round(0.5773502692 * 32768) = 18919`
- `SQRT3_OVER_2   = round(0.8660254038 * 32768) = 28378`
- `HALF_Q15       = 16384` (= 0.5)

## Transforms (exact arithmetic)

**Clarke** (ia, ib in LSB; balanced ⇒ ic = −ia−ib):
```
ialpha = ia
ibeta  = ((ia + 2*ib) * ONE_OVER_SQRT3) >>> 15
```

**Park** (rotation by θe; c = cos, s = sin in Q15):
```
id =  ( ialpha*c + ibeta*s) >>> 15
iq =  (-ialpha*s + ibeta*c) >>> 15
```

**Inverse Park** (vd, vq in duty units):
```
valpha = (vd*c - vq*s) >>> 15
vbeta  = (vd*s + vq*c) >>> 15
```

**sin/cos LUT**: a quarter-wave table of `2^sincos_table_bits` entries holds
sin over [0, π/2) in Q15. The 16-bit angle's top 2 bits select the quadrant;
the next `sincos_table_bits` bits index the table (cos = sin(θ+π/2), same
table with the quadrant offset). Reference and RTL build the table from the
same rounding rule: `table[i] = round(sin((i+0.5)/2^bits * π/2) * 32767)`.

## Current PI (per axis, parallel form, conditional-integration anti-windup)

```
err   = i_target - i_meas                      // LSB
integ = integ + err          (unless saturated and pushing further in)
v_raw = KP*err + (KP*integ >>> KI_SHIFT)        // duty units
```
`KP = foc.cur_pi_kp`, `KI_SHIFT = foc.cur_pi_ki_shift`. Output is bounded by
the **voltage-circle limiter** applied to (vd, vq) jointly, not per axis.

## Voltage-circle limiter

The realizable inscribed-circle radius in these units is `V_MAX = HALF`
(phase peak at the SVPWM inscribed circle). The limiter caps the *vector*:
```
VLIM = (V_MAX * v_circle_limit)          // duty units, leaves headroom
mag2 = vd*vd + vq*vq
if mag2 > VLIM*VLIM:   scale vd, vq by VLIM/sqrt(mag2)   // priority: vd
```
RTL uses one reciprocal-sqrt step (or an iterative shrink); the reference
uses exact sqrt, and the parity test allows a small tolerance on the limiter
boundary only.

## SVPWM = min/max common-mode injection

Phase references from (vα, vβ), in duty units centered at zero:
```
ra = valpha
rb = (-valpha + ((SQRT3_OVER_2*vbeta) >>> 15) * 2 ... )   // see reference
   // exactly: rb = (-valpha + sqrt3*vbeta)/2, rc = (-valpha - sqrt3*vbeta)/2
```
implemented as
```
half_alpha = valpha >>> 1
s3b        = (SQRT3_OVER_2 * vbeta) >>> 15      // = (sqrt3/2)*vbeta
ra =  valpha
rb = -half_alpha + s3b
rc = -half_alpha - s3b
cm   = (max(ra,rb,rc) + min(ra,rb,rc)) >>> 1     // common-mode injection
duty[k] = sat_duty( HALF_HALF + (r[k] - cm) )    // HALF_HALF = HALF>>1
```
`HALF_HALF` (= 50% duty) is the zero-voltage operating point; the injected
references swing each leg around it. The line-to-line voltages are unchanged
by `cm`, which is the whole point — it buys ~15% more linear range than pure
sine PWM while keeping every leg in [0, HALF].

## Why LSB-native control

Working in ADC LSB (not amps) means the current loop needs no physical
scaling constants, so it is immune to the still-unmeasured shunt/gain values
(Q7) — only the *gains* `cur_pi_kp/ki_shift` carry tuning, and those are
already Q1-flagged. Torque-linearity and dq-tracking assertions are made in
LSB and in plant-truth amps side by side, so the LSB-native choice is checked
against physical units, not hidden by it.
