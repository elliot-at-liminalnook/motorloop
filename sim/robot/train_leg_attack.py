# SPDX-License-Identifier: MIT
"""Train the runtime-selectable FL/FR/RL/RR attack-and-kick curriculum."""

from __future__ import annotations

import sys

from warp_train_cli import run


if __name__ == "__main__":
    run("leg_attack", sys.argv[1:], default_tag="leg_attack_warp")
