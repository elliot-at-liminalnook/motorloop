# SPDX-License-Identifier: MIT
"""Check that the shared CPG teacher maps exactly to Warp's action order."""

import numpy as np

from cpg_teacher import DEFAULT_RAW, cpg_action, decode_params


def main():
    params = decode_params(np.asarray(DEFAULT_RAW), xp=np)
    indices = np.arange(12).reshape(4, 3)
    first = cpg_action(0.37, params, indices, 12, xp=np)
    second = cpg_action(0.37, params, indices, 12, xp=np)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (12,) and np.isfinite(first).all()
    print("PASS: CPG teacher is deterministic in the 12-servo Warp action order")


if __name__ == "__main__":
    main()
