# SPDX-License-Identifier: MIT
"""Manifest for skill-specific fighter deltas.

The bank records checkpoints that are useful for a scoped skill but not yet safe
to merge into the shared fighter. This keeps continual-learning decisions
explicit: a checkpoint can be a bridge/clean/etc. adapter candidate even when the
global robust selector rejects it for shared promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))
DEFAULT_BANK = OUT / "skill_bank.json"


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: str | Path | None):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_bank(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"version": 1, "entries": []}


def save_bank(path: Path, bank: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    bank["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(bank, indent=2, sort_keys=True))


def selector_summary(selector: dict | None) -> dict | None:
    if not selector:
        return None
    accepted = selector.get("accepted")
    cand = (selector.get("candidates") or [{}])[0]
    return {
        "accepted": accepted is not None,
        "slices": selector.get("slices", []),
        "seeds": selector.get("seeds", []),
        "summary": cand.get("summary", {}),
        "mean_margin_delta": cand.get("mean_margin_delta"),
        "mean_judge_delta": cand.get("mean_judge_delta"),
        "checks": cand.get("checks", {}),
    }


def cmd_add(args):
    bank_path = Path(args.bank)
    bank = load_bank(bank_path)
    delta = load_json(args.delta_json)
    selector = load_json(args.selector_json)
    ckpt = Path(args.ckpt)
    entry = {
        "name": args.name,
        "skill": args.skill,
        "scope": args.scope,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ckpt": str(ckpt),
        "ckpt_sha256": sha256(ckpt),
        "baseline": str(args.baseline or ""),
        "opponent": str(args.opponent or ""),
        "delta_json": str(args.delta_json or ""),
        "delta_npz": str(args.delta_npz or ""),
        "selector_json": str(args.selector_json or ""),
        "delta_decision": (delta or {}).get("decision", {}),
        "delta_relative_norm": (delta or {}).get("relative_norm"),
        "selector": selector_summary(selector),
        "notes": args.notes,
    }
    entries = [e for e in bank.get("entries", []) if e.get("name") != args.name]
    entries.append(entry)
    bank["entries"] = entries
    save_bank(bank_path, bank)
    print(json.dumps(entry, indent=2, sort_keys=True))


def cmd_list(args):
    bank = load_bank(Path(args.bank))
    entries = bank.get("entries", [])
    if args.skill:
        entries = [e for e in entries if e.get("skill") == args.skill]
    if args.scope:
        entries = [e for e in entries if e.get("scope") == args.scope]
    if args.json:
        print(json.dumps({"entries": entries}, indent=2, sort_keys=True))
        return
    for e in entries:
        sel = e.get("selector") or {}
        print(
            f"{e.get('name')} skill={e.get('skill')} scope={e.get('scope')} "
            f"ckpt={Path(e.get('ckpt', '')).name} "
            f"selector_accepted={sel.get('accepted')} "
            f"mean_margin_delta={sel.get('mean_margin_delta')} "
            f"mean_judge_delta={sel.get('mean_judge_delta')}"
        )


def cmd_resolve(args):
    bank = load_bank(Path(args.bank))
    candidates = [
        e for e in bank.get("entries", [])
        if e.get("skill") == args.skill and e.get("scope") in args.scope
    ]
    if not candidates:
        raise SystemExit(f"no entry for skill={args.skill} scope={','.join(args.scope)}")
    def score(e):
        sel = e.get("selector") or {}
        return float(sel.get("mean_judge_delta") or 0.0), float(sel.get("mean_margin_delta") or 0.0)
    best = max(candidates, key=score)
    print(best["ckpt"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=str(DEFAULT_BANK))
    sub = ap.add_subparsers(dest="cmd", required=True)

    add = sub.add_parser("add")
    add.add_argument("--name", required=True)
    add.add_argument("--skill", required=True)
    add.add_argument("--scope", choices=["shared", "skill", "candidate", "rejected"], required=True)
    add.add_argument("--ckpt", required=True)
    add.add_argument("--baseline", default="")
    add.add_argument("--opponent", default="")
    add.add_argument("--delta-json", default="")
    add.add_argument("--delta-npz", default="")
    add.add_argument("--selector-json", default="")
    add.add_argument("--notes", default="")
    add.set_defaults(func=cmd_add)

    ls = sub.add_parser("list")
    ls.add_argument("--skill", default="")
    ls.add_argument("--scope", default="")
    ls.add_argument("--json", action="store_true")
    ls.set_defaults(func=cmd_list)

    resolve = sub.add_parser("resolve")
    resolve.add_argument("--skill", required=True)
    resolve.add_argument("--scope", action="append", default=["skill", "shared"])
    resolve.set_defaults(func=cmd_resolve)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
