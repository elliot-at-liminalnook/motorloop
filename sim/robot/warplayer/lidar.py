# SPDX-License-Identifier: MIT
"""warplayer.lidar — component (iii) of the thin bespoke layer (§10c).

A 144-ray x nworld lidar kernel over analytic primitives, one thread per
(world, ray), attached to the robot's torso BODY frame. The ray casts REUSE
mujoco_warp's analytic intersection library (`ray_geom` dispatching to
ray_plane / ray_sphere / ray_capsule / ray_box, mujoco_warp/_src/ray.py:187,
211, 228, 397, dispatch :808-828) — importing the @wp.func objects directly
guarantees bit-parity with the engine's own rangefinder instead of a
transcription of the math.

Semantics mirror MuJoCo's rangefinder sensor exactly:
  * ray origin  = site position   = torso_xpos + torso_xmat @ origin_local
  * ray dir     = site +z axis    = torso_xmat @ dir_local
    (mujoco_warp/_src/sensor.py:179-197 `_sensor_rangefinder_init`)
  * the site's own body is excluded (sensor.py:840 passes
    `sensor_rangefinder_bodyid` as bodyexclude to ray.rays)
  * miss -> -1.0; hit -> min(dist, cutoff) — the positive-datatype sensor
    cutoff clamp (mujoco_warp/_src/sensor.py:78 `out[adr] = wp.min(...)`)

The invisible-geom exclusion of the full `_ray_eliminate` (ray.py:52-99) is
NOT replicated: the fight scene gives every geom material mat0 with alpha 1
(gen_robot_mjcf.py:242-248), so nothing is ever eliminated by visibility.
This is asserted host-side in `lidar_tables_from_model`.

mujoco_warp Data fields read IN-PLACE by the kernel (graph-capture ready —
no host round-trip, no host branches, fixed launch dims):
  d.xpos      (nworld, nbody)  torso frame origin
  d.xmat      (nworld, nbody)  torso frame orientation
  d.geom_xpos (nworld, ngeom)  geom poses for the intersections
  d.geom_xmat (nworld, ngeom)
Model fields: m.geom_type (ngeom,), m.geom_bodyid (ngeom,),
m.geom_size (*, ngeom) broadcastable — indexed worldid % leading dim, the
per-world-DR convention (§4).
"""
from __future__ import annotations

import numpy as np
import warp as wp
from mujoco_warp._src.ray import ray_geom  # analytic primitive intersections (see module docstring)


@wp.kernel
def lidar_kernel(
    # Model:
    geom_type: wp.array(dtype=wp.int32),
    geom_bodyid: wp.array(dtype=wp.int32),
    geom_size: wp.array2d(dtype=wp.vec3),
    # Data (read in-place):
    xpos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_xmat: wp.array2d(dtype=wp.mat33),
    # In (static tables):
    torso_body: int,
    origin_local: wp.vec3,
    dirs_local: wp.array(dtype=wp.vec3),
    max_range: float,
    # Out:
    dist_out: wp.array2d(dtype=wp.float32),   # raw rangefinder value: -1 miss, else min(d, max_range)
    scan_out: wp.array2d(dtype=wp.float32),   # normalized [0,1] scan (train_adversarial._lidar_scan:507-517)
):
    worldid, rayid = wp.tid()
    R = xmat[worldid, torso_body]
    pnt = xpos[worldid, torso_body] + R @ origin_local
    vec = R @ dirs_local[rayid]

    best = float(-1.0)
    mw = geom_size.shape[0]
    for g in range(geom_type.shape[0]):
        if geom_bodyid[g] == torso_body:      # bodyexclude: the site's own body
            continue
        dist, _ = ray_geom(
            geom_xpos[worldid, g],
            geom_xmat[worldid, g],
            geom_size[worldid % mw, g],
            pnt,
            vec,
            geom_type[g],
        )
        if dist >= 0.0 and (best < 0.0 or dist < best):
            best = dist

    if best >= 0.0:
        best = wp.min(best, max_range)        # positive-sensor cutoff clamp
    dist_out[worldid, rayid] = best
    # normalized scan exactly as the env consumes it: miss -> 1.0 (= max range)
    hit = wp.where(best < 0.0, max_range, wp.clamp(best, 0.0, max_range))
    scan_out[worldid, rayid] = hit / max_range


def lidar_tables_from_model(mjm, prefix: str = "A_lidar") -> tuple[int, np.ndarray, np.ndarray]:
    """Extract (torso_bodyid, origin_local(3,), dirs_local(nray,3)) from the
    lidar-enabled MjModel's site tables (gen_robot_mjcf._lidar_sites_xml:300-358
    puts all `A_lidar_*` sites at one local pos on A_torso, ray = site +z).

    Site order is site-id order == the sensor/sensordata order (the generator
    emits one rangefinder per site in the same sequence)."""
    import mujoco

    sids = [s for s in range(mjm.nsite)
            if (mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_SITE, s) or "").startswith(prefix)]
    if not sids:
        raise ValueError(f"no '{prefix}*' sites — build the scene with lidar=True")
    bodies = {int(mjm.site_bodyid[s]) for s in sids}
    if len(bodies) != 1:
        raise ValueError(f"lidar sites span bodies {bodies}; expected one torso")
    origins = mjm.site_pos[sids]
    if not np.allclose(origins, origins[0]):
        raise ValueError("lidar sites at differing local offsets; kernel assumes one origin")
    # visibility precondition for skipping _ray_eliminate's material checks
    vis = np.where(mjm.geom_matid >= 0, mjm.mat_rgba[mjm.geom_matid, 3], mjm.geom_rgba[:, 3])
    if (vis == 0.0).any():
        raise ValueError("scene has invisible geoms; kernel skips the visibility exclusion")
    dirs = np.empty((len(sids), 3), dtype=np.float64)
    for i, s in enumerate(sids):
        w, x, y, z = mjm.site_quat[s]          # mujoco order (w, x, y, z)
        # dir = R(q) @ (0,0,1), the rangefinder's +z convention
        dirs[i] = (2.0 * (x * z + w * y),
                   2.0 * (y * z - w * x),
                   1.0 - 2.0 * (x * x + y * y))
    return int(next(iter(bodies))), origins[0].astype(np.float32), dirs.astype(np.float32)


class Lidar:
    """Host-side wrapper owning the static tables + output buffers for one robot."""

    def __init__(self, mjm, nworld: int, max_range: float = 2.0, prefix: str = "A_lidar"):
        body, origin, dirs = lidar_tables_from_model(mjm, prefix)
        self.torso_body = body
        self.nray = dirs.shape[0]
        self.max_range = float(max_range)
        self.origin_local = wp.vec3(*origin)
        self.dirs_local = wp.array(dirs, dtype=wp.vec3)
        self.dist = wp.zeros((nworld, self.nray), dtype=wp.float32)
        self.scan = wp.zeros((nworld, self.nray), dtype=wp.float32)

    def launch(self, m, d):
        """Append the lidar to the step's launch sequence (graph-capturable)."""
        wp.launch(
            lidar_kernel,
            dim=(d.nworld, self.nray),
            inputs=[m.geom_type, m.geom_bodyid, m.geom_size,
                    d.xpos, d.xmat, d.geom_xpos, d.geom_xmat,
                    self.torso_body, self.origin_local, self.dirs_local, self.max_range],
            outputs=[self.dist, self.scan],
        )
