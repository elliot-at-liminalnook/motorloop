# SPDX-License-Identifier: MIT
"""Typed validation of the robot spec (robot.toml) — pydantic v2 gate at model build.

A malformed/nonsense spec should die when the MJCF is generated, not 40 minutes
into a training run. `load_spec()` (gen_robot_mjcf.py) calls `validate_spec()` on
the freshly-loaded dict; the models below encode the physical contract the
generator and envs assume:

  * positive lengths/radii/masses, positive gear/peak_factor
  * joint ranges are (lo < hi) pairs; the knee is ONE-WAY (hi < 0 — bends one way)
  * stand_* reference angles sit inside their joint ranges
  * every domain_randomization bracket [lo, hi] contains its spec center
    (gear bracket ∋ actuator.gear, joint_stiffness bracket ∋
    leg_defaults.joint_stiffness, torso_mass bracket ∋ torso.mass, …)
  * spawn_height clears the kinematic stance height
    thigh_len·cos(stand_flex) + calf_len·cos(stand_flex + stand_knee)
    (spawn above the standing leg extension, not inside the floor)

Unknown keys are ALLOWED but warned about (extra="allow") — the spec is a living
config; a typo'd key should shout, not crash a pod that ships an older schema.

`validate_spec(d)` returns the ORIGINAL dict: callers keep using plain dicts
everywhere — validation is a gate, not a type migration.
"""
from __future__ import annotations

import math
import warnings
from typing import Annotated, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PosFloat = Annotated[float, Field(gt=0)]
NonNegFloat = Annotated[float, Field(ge=0)]
Pair = Tuple[float, float]
Vec3 = Tuple[float, float, float]


class _SpecModel(BaseModel):
    """Base: unknown keys allowed (collected in model_extra, warned by validate_spec)."""
    model_config = ConfigDict(extra="allow")


class MetaSpec(_SpecModel):
    name: str = Field(min_length=1)


class TorsoSpec(_SpecModel):
    half_extents: Tuple[PosFloat, PosFloat, PosFloat]
    mass: PosFloat
    spawn_height: PosFloat


class LegDefaultsSpec(_SpecModel):
    thigh_len: PosFloat
    calf_len: PosFloat
    link_radius: PosFloat
    hip_offset: NonNegFloat
    hip_mass: PosFloat = 0.3
    thigh_mass: PosFloat
    calf_mass: PosFloat
    foot_radius: PosFloat
    foot_mass: PosFloat = 0.05
    # stand_* defaults mirror gen_robot_mjcf._leg_xml's .get() fallbacks.
    stand_abd: float = 0.0
    stand_flex: float = -0.4
    stand_knee: float = -1.1
    abd_range: Pair
    flex_range: Pair
    knee_range: Pair
    joint_damping: NonNegFloat = 0.0
    joint_stiffness: NonNegFloat = 0.0

    @field_validator("abd_range", "flex_range", "knee_range")
    @classmethod
    def _lo_lt_hi(cls, v: Pair, info) -> Pair:
        lo, hi = v
        if not lo < hi:
            raise ValueError(f"{info.field_name} must be a (lo < hi) pair, got [{lo}, {hi}]")
        return v

    @field_validator("knee_range")
    @classmethod
    def _knee_one_way(cls, v: Pair) -> Pair:
        lo, hi = v
        if hi >= 0:
            raise ValueError(
                f"knee_range must be one-way (hi < 0; the knee bends one way), got [{lo}, {hi}]")
        return v

    @model_validator(mode="after")
    def _stand_within_ranges(self) -> "LegDefaultsSpec":
        for stand, rng in (("stand_abd", "abd_range"),
                           ("stand_flex", "flex_range"),
                           ("stand_knee", "knee_range")):
            val, (lo, hi) = getattr(self, stand), getattr(self, rng)
            if not lo <= val <= hi:
                raise ValueError(
                    f"{stand}={val} outside {rng}=[{lo}, {hi}] — the standing reference "
                    f"angle must be reachable")
        return self


class ActuatorSpec(_SpecModel):
    motor: str = Field(min_length=1)
    peak_factor: PosFloat
    gear: PosFloat
    voltage: Optional[PosFloat] = None


class ContactSpec(_SpecModel):
    friction: Optional[Tuple[NonNegFloat, NonNegFloat, NonNegFloat]] = None
    solref: Optional[Pair] = None
    solimp: Optional[Tuple[float, ...]] = None
    floor_calf_solref: Optional[Pair] = None
    floor_calf_solimp: Optional[Tuple[float, ...]] = None
    calf_floor: Optional[bool] = None
    disable_calf_floor: Optional[bool] = None


class StrikerSpec(_SpecModel):
    enabled: bool = False
    stroke: Optional[PosFloat] = None
    rod_len: Optional[PosFloat] = None
    rod_radius: Optional[PosFloat] = None
    rod_density: Optional[PosFloat] = None
    bore: Optional[PosFloat] = None
    pressure: Optional[PosFloat] = None
    valve_tau: Optional[PosFloat] = None
    return_stiffness: Optional[NonNegFloat] = None
    fire_cost: Optional[NonNegFloat] = None

    @model_validator(mode="after")
    def _enabled_needs_geometry(self) -> "StrikerSpec":
        if self.enabled:
            missing = [k for k in ("stroke", "rod_len", "rod_radius", "rod_density",
                                   "bore", "pressure", "valve_tau", "return_stiffness")
                       if getattr(self, k) is None]
            if missing:
                raise ValueError(f"striker.enabled=true but missing: {', '.join(missing)}")
        return self


class LegSpec(_SpecModel):
    name: str = Field(min_length=1)
    pos: Vec3
    is_weapon: bool = False


class DomainRandomizationSpec(_SpecModel):
    torso_mass: Optional[Pair] = None
    thigh_len: Optional[Pair] = None
    calf_len: Optional[Pair] = None
    gear: Optional[Pair] = None
    joint_stiffness: Optional[Pair] = None

    @field_validator("torso_mass", "thigh_len", "calf_len", "gear", "joint_stiffness")
    @classmethod
    def _bracket_ordered(cls, v: Optional[Pair], info) -> Optional[Pair]:
        if v is not None:
            lo, hi = v
            if not lo <= hi:
                raise ValueError(f"{info.field_name} bracket must be [lo <= hi], got [{lo}, {hi}]")
        return v


# DR bracket -> the spec scalar it randomizes around (the center it must contain).
_DR_CENTERS = {
    "torso_mass": lambda s: s.torso.mass,
    "thigh_len": lambda s: s.leg_defaults.thigh_len,
    "calf_len": lambda s: s.leg_defaults.calf_len,
    "gear": lambda s: s.actuator.gear,
    "joint_stiffness": lambda s: s.leg_defaults.joint_stiffness,
}


class RobotSpec(_SpecModel):
    meta: MetaSpec
    torso: TorsoSpec
    leg_defaults: LegDefaultsSpec
    actuator: ActuatorSpec
    contact: Optional[ContactSpec] = None
    striker: Optional[StrikerSpec] = None
    leg: list[LegSpec] = Field(min_length=1)
    domain_randomization: Optional[DomainRandomizationSpec] = None

    @model_validator(mode="after")
    def _dr_brackets_contain_centers(self) -> "RobotSpec":
        if self.domain_randomization is None:
            return self
        for key, center_of in _DR_CENTERS.items():
            bracket = getattr(self.domain_randomization, key)
            if bracket is None:
                continue
            lo, hi = bracket
            center = center_of(self)
            if not lo <= center <= hi:
                raise ValueError(
                    f"domain_randomization.{key} bracket [{lo}, {hi}] does not contain the "
                    f"spec center {center} — DR must randomize AROUND the nominal value")
        return self

    @model_validator(mode="after")
    def _spawn_above_stance(self) -> "RobotSpec":
        d = self.leg_defaults
        stance = (d.thigh_len * math.cos(d.stand_flex)
                  + d.calf_len * math.cos(d.stand_flex + d.stand_knee))
        if self.torso.spawn_height <= stance:
            raise ValueError(
                f"torso.spawn_height={self.torso.spawn_height} must exceed the kinematic "
                f"stance height {stance:.4f} m (thigh_len*cos(stand_flex) + "
                f"calf_len*cos(stand_flex+stand_knee)) — the body would spawn with the "
                f"standing legs through the floor")
        return self


def _warn_unknown_keys(model: BaseModel, path: str = "") -> None:
    """Recursively warn on extra keys (allowed but not part of the validated contract)."""
    for key in (model.model_extra or {}):
        loc = f"{path}.{key}" if path else key
        warnings.warn(f"robot spec: unknown key '{loc}' (allowed, but not validated)",
                      stacklevel=2)
    for name in type(model).model_fields:
        val = getattr(model, name, None)
        sub = f"{path}.{name}" if path else name
        if isinstance(val, BaseModel):
            _warn_unknown_keys(val, sub)
        elif isinstance(val, list):
            for i, item in enumerate(val):
                if isinstance(item, BaseModel):
                    _warn_unknown_keys(item, f"{sub}[{i}]")


def validate_spec(d: dict) -> dict:
    """Validate a robot spec dict; raise pydantic.ValidationError on a nonsense spec.

    Returns the ORIGINAL dict unchanged — callers use plain dicts everywhere;
    this is a gate at model build, not a type migration.
    """
    model = RobotSpec.model_validate(d)
    _warn_unknown_keys(model)
    return d
