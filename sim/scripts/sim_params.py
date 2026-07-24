#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Loader for sim/config/params.toml with provenance-aware assumption flags.

Every simulation parameter carries a `status` describing how trustworthy it
is. Any run that consumes parameters with an unconfirmed status (see
[meta].unconfirmed_statuses in the config) must:

  1. print the assumption banner before producing results, and
  2. write the same banner as a sidecar file next to every output artifact,

so no trace can be mistaken for a prediction about the real hardware.

Standard library only (tomllib requires Python >= 3.11).
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "params.toml"

_BLOCKED_BY_PATTERN = re.compile(r"^Q\d+$")
_REQUIRED_KEYS = {"value", "unit", "status"}
_OPTIONAL_KEYS = {"blocked_by", "source", "note", "derived_from"}


class ParamConfigError(Exception):
    """Raised when params.toml violates the parameter-table convention."""


@dataclass(frozen=True)
class ParamEntry:
    path: str
    value: object
    unit: str
    status: str
    blocked_by: str | None = None
    source: str | None = None
    note: str | None = None
    derived_from: str | None = None


@dataclass(frozen=True)
class SimParams:
    config_path: Path
    statuses: list[str]
    unconfirmed_statuses: list[str]
    entries: dict[str, ParamEntry] = field(default_factory=dict)

    def value(self, path: str) -> object:
        return self.entries[path].value

    def derived_entries(self) -> list[ParamEntry]:
        """Parameters carrying a derived_from reference, validated to point
        at an existing circuit/motor-spec table."""
        return [e for e in self.entries.values() if e.derived_from]

    def circuit_values(self, table: str) -> dict[str, object]:
        """All component values under e.g. 'circuit.emf_channel'."""
        prefix = table + "."
        return {
            e.path[len(prefix):]: e.value
            for e in self.entries.values()
            if e.path.startswith(prefix)
        }

    def unconfirmed(self) -> list[ParamEntry]:
        order = {status: rank for rank, status in enumerate(self.statuses)}
        flagged = [
            entry
            for entry in self.entries.values()
            if entry.status in self.unconfirmed_statuses
        ]
        return sorted(flagged, key=lambda e: (-order.get(e.status, 0), e.path))

    def banner_text(self) -> str:
        flagged = self.unconfirmed()
        bar = "=" * 78
        lines = [bar]
        if not flagged:
            lines.append("  All simulation parameters are confirmed (measured/datasheet/decided).")
        else:
            lines.append(
                f"  !! UNCONFIRMED ASSUMPTIONS: {len(flagged)} parameter(s) !!"
            )
            lines.append(f"  config: {self.config_path}")
            lines.append("-" * 78)
            for entry in flagged:
                blocked = f"  ({entry.blocked_by})" if entry.blocked_by else ""
                lines.append(
                    f"  [{entry.status:<15}] {entry.path} = {entry.value} {entry.unit}{blocked}"
                )
            lines.append("-" * 78)
            lines.append(
                "  Results below are NOT hardware predictions. Resolve via"
                " notes/open-questions.md."
            )
        lines.append(bar)
        return "\n".join(lines)

    def write_sidecar(self, artifact_path: Path) -> Path:
        sidecar = artifact_path.with_suffix(artifact_path.suffix + ".assumptions.txt")
        sidecar.write_text(self.banner_text() + "\n")
        return sidecar


def _is_param_table(node: object) -> bool:
    return isinstance(node, dict) and "value" in node


def _validate_entry(path: str, table: dict, statuses: list[str], unconfirmed: list[str]) -> ParamEntry:
    missing = _REQUIRED_KEYS - table.keys()
    if missing:
        raise ParamConfigError(f"{path}: missing required key(s) {sorted(missing)}")
    unknown = table.keys() - _REQUIRED_KEYS - _OPTIONAL_KEYS
    if unknown:
        raise ParamConfigError(f"{path}: unknown key(s) {sorted(unknown)}")
    status = table["status"]
    if status not in statuses:
        raise ParamConfigError(f"{path}: status '{status}' not in {statuses}")
    blocked_by = table.get("blocked_by")
    if blocked_by is not None and not _BLOCKED_BY_PATTERN.match(blocked_by):
        raise ParamConfigError(f"{path}: blocked_by '{blocked_by}' must look like 'Q12'")
    if status in unconfirmed and blocked_by is None and "note" not in table:
        raise ParamConfigError(
            f"{path}: unconfirmed status '{status}' needs a blocked_by or note"
        )
    derived_from = table.get("derived_from")
    if derived_from is not None and not (
        derived_from == "motor_spec" or derived_from.startswith("circuit.")
    ):
        raise ParamConfigError(
            f"{path}: derived_from '{derived_from}' must name a "
            f"[circuit.*] table or motor_spec"
        )
    return ParamEntry(
        path=path,
        value=table["value"],
        unit=table["unit"],
        status=status,
        blocked_by=blocked_by,
        source=table.get("source"),
        note=table.get("note"),
        derived_from=derived_from,
    )


def _walk(node: dict, prefix: str, statuses: list[str], unconfirmed: list[str], entries: dict[str, ParamEntry]) -> None:
    for key, child in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if _is_param_table(child):
            entries[path] = _validate_entry(path, child, statuses, unconfirmed)
        elif isinstance(child, dict):
            _walk(child, path, statuses, unconfirmed, entries)
        else:
            raise ParamConfigError(
                f"{path}: bare value not allowed; use a parameter table with"
                " value/unit/status"
            )


def load(config_path: Path | None = None) -> SimParams:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise ParamConfigError(f"config not found: {path}")
    with path.open("rb") as f:
        raw = tomllib.load(f)

    meta = raw.pop("meta", None)
    if not meta or "statuses" not in meta or "unconfirmed_statuses" not in meta:
        raise ParamConfigError(
            "[meta] with 'statuses' and 'unconfirmed_statuses' is required"
        )
    statuses = list(meta["statuses"])
    unconfirmed = list(meta["unconfirmed_statuses"])
    bad = set(unconfirmed) - set(statuses)
    if bad:
        raise ParamConfigError(f"[meta]: unconfirmed statuses {sorted(bad)} not in statuses")

    entries: dict[str, ParamEntry] = {}
    _walk(raw, "", statuses, unconfirmed, entries)

    # derived_from references must point at tables that actually exist.
    for entry in entries.values():
        if entry.derived_from:
            prefix = entry.derived_from + "."
            if not any(p.startswith(prefix) for p in entries):
                raise ParamConfigError(
                    f"{entry.path}: derived_from references missing table "
                    f"[{entry.derived_from}]"
                )

    return SimParams(
        config_path=path,
        statuses=statuses,
        unconfirmed_statuses=unconfirmed,
        entries=entries,
    )


if __name__ == "__main__":
    params = load()
    print(params.banner_text())
    print(f"\n{len(params.entries)} parameters loaded, "
          f"{len(params.unconfirmed())} unconfirmed.")
