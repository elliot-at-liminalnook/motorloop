<!-- SPDX-License-Identifier: MIT -->
# Blender and Phobos agent workflow

> **Document status:** Current · **Audience:** Agents and robot-model contributors · **Last reviewed:** 2026-07-12 · **Canonical for:** Agent-mediated Blender and Phobos interaction on the configured GNOME workstation

This runbook describes the safe interaction pattern established while building
the planar slider-crank model. Follow it when inspecting Blender, guiding a new
Blender user, making live model changes, taking screenshots, or recovering a
session after Blender or the computer restarts.

The intended outcome is a collaborative loop in which the agent can inspect and
modify Blender through a narrow application programming interface (API), the
user remains able to see and override every change, and each material mutation
is backed up and numerically or visually checked.

## Operating rules

1. Inspect the running process, active file, dirty state, scene, selection, and
   relevant objects before changing anything.
2. Use `blender-control` for Blender mutations. Do not use general desktop input
   automation or try to drive Blender by screen coordinates.
3. Use `blender-view` only for view-only screenshots. It never grants input
   control.
4. Preserve existing work. Save a new model under a new filename, and never
   overwrite the tutorial file merely because it is open.
5. Make one coherent change, then query or validate the actual Blender state.
   A plausible-looking viewport is not sufficient evidence.
6. Keep the user informed before tool-driven changes and during work that lasts
   more than about a minute.
7. For tutorial guidance, give two or three actions at a time. Explain the
   purpose of each action and its physical consequence for the mechanism.
8. The user is new to Blender and accesses the workstation from a Mac. Prefer
   visible menus and named controls over unexplained shortcuts. Do not assume
   an F3 key is available.

## Installed components

| Component | Location | Purpose |
| --- | --- | --- |
| Blender/Phobos launcher | `~/.local/bin/blender-phobos` | Starts Blender 4.2.22 with Phobos' Python modules on the correct path |
| Live-control client | `~/.local/bin/blender-control` | Sends structured, allowlisted requests to the active Blender process |
| Live-control add-on | `~/.config/blender/4.2/scripts/addons/codex_blender_control/` | Runs Blender-side operations on Blender's main thread |
| Slider-crank builder | `~/.config/blender/4.2/scripts/addons/codex_blender_control/slider_crank.py` | Builds and verifies the project-specific parametric linkage |
| View-only bridge | `~/.local/bin/blender-view` | Captures the selected Blender window through the GNOME ScreenCast portal |
| Guidance overlay | `~/.local/bin/blender-overlay` | Draws click-through annotations relative to Blender when window tracking is valid |

The current mechanism is
[`Slider_Crank_Toggle.blend`](../Slider_Crank_Toggle.blend). The earlier
[`Phobos_Tutorial_Arm.blend`](../Phobos_Tutorial_Arm.blend) is a separate file
and should remain intact.

## First-contact checklist

Start every Blender turn with read-only checks:

```bash
blender-control status
blender-view status
blender-overlay status
```

From `blender-control status`, confirm:

- `status` is `ready`;
- `file` is the file the user expects;
- `dirty` is understood before any restart or file switch;
- the listed Blender process is the one that should receive changes; and
- the requested command appears in `capabilities`.

Then inspect the model before relying on remembered names:

```bash
blender-control scene --properties
blender-control object SliderCrankController
```

`scene --properties` currently summarizes all Blender object datablocks, which
can include objects belonging to another scene in the same `.blend`. Treat its
scene name, active object, selection, and per-object details as authoritative;
do not treat the global object count as an active-scene count.

If the control socket is not running, do not immediately start a second Blender
instance. First check for an existing process:

```bash
ps -eo pid,ppid,stat,cmd | rg -i 'blender|codex-blender-view' | rg -v 'rg -i'
```

Only one Blender process owns the live-control socket. An accidentally launched
second process will remain in standby and will not receive commands.

## Starting or restoring Blender

Use the Phobos-aware launcher in a persistent terminal or PTY:

```bash
blender-phobos /home/elliot/Projects/bldc-cosim-testbench/Slider_Crank_Toggle.blend
```

Poll `blender-control status` until it reports `ready`. The control add-on is
enabled persistently and starts its Unix-domain socket when Blender loads.

For the original tutorial instead:

```bash
blender-phobos /home/elliot/Projects/bldc-cosim-testbench/Phobos_Tutorial_Arm.blend
```

After an unexpected shutdown:

1. Check the target `.blend` and the automatic backup directory described
   below.
2. Open the intended file with `blender-phobos`.
3. Confirm the active file and dirty state with `blender-control status`.
4. For the slider-crank, immediately run `blender-control slider-crank-verify`.
5. Start the view bridge again if screenshots are needed.

## Safe live-control loop

### Read before writing

Useful read-only commands are:

```bash
blender-control status
blender-control scene --properties
blender-control object OBJECT_NAME
blender-control slider-crank-verify
```

The normal mutation loop is:

1. State the intended change and physical reason to the user.
2. Query the target object and current selection.
3. Send one structured mutation.
4. Check the response has `ok: true` and note the returned backup path.
5. Query the affected object or run the model-specific validator.
6. Take a screenshot when spatial interpretation matters.
7. Save only after the state is correct.

Examples of structured mutations:

```bash
blender-control select base_link --active base_link
blender-control transform OBJECT_NAME --location 0 0 0
blender-control pose ARMATURE_NAME --bone BONE_NAME --rotation-deg 0 0 20
blender-control property OBJECT_NAME joint/type '"revolute"'
blender-control parent PARENT_NAME CHILD_NAME
blender-control undo
blender-control redo
```

Run `blender-control COMMAND --help` before using an unfamiliar command. Prefer
these command-specific interfaces over the raw `request` interface. The raw
interface still has an allowlist and deliberately provides no Python `eval`.

When several transforms must change together, a `batch` request can group up to
50 allowlisted operations behind one backup/undo boundary. Do not use a batch
merely to save round trips when intermediate validation would be valuable.

### Selection and mode are state

Blender operations frequently distinguish "selected" from "active." Never say
only "select these two objects" when order matters. Name the order and which
object must become active. For the tutorial step already encountered, the user
needed to click `base_link` first and then Shift-click `base_visual`; future
instructions should preserve exact ordering rather than infer it from the final
highlight.

Also query or explicitly set Object, Edit, or Pose mode before operations whose
meaning changes by mode.

### Save, undo, and backups

Every model-changing live-control request automatically:

- writes a compressed `.blend` checkpoint;
- brackets direct Blender property writes with undo snapshots; and
- appends the request and result to a private audit log.

The locations are:

```text
~/.local/state/codex-blender-control/backups/
~/.local/state/codex-blender-control/audit.jsonl
~/.local/state/codex-blender-control/state.json
```

The bridge retains the newest 50 automatic backups. Backups reduce risk but do
not replace deliberate saves:

```bash
blender-control save
blender-control save --filepath /home/elliot/Projects/bldc-cosim-testbench/NEW_MODEL.blend
```

Before restarting Blender, confirm `dirty: false`. If Blender must be terminated,
send a normal termination signal only after saving; do not use `kill -9` for a
routine restart.

If the add-on source itself changes, run `python3 -m py_compile` on the changed
module and restart Blender. The already imported Python module is not reliably
hot-reloaded in the live process.

## View-only screenshots

Start the bridge and ask the user to select the Blender window in GNOME's share
dialog when required:

```bash
blender-view start
blender-view status
```

A usable status reports `ready`, `view_only: true`, and
`input_control: false`. Capture a PNG with:

```bash
blender-view shot /home/elliot/Pictures/blender-check.png
```

An agent should then inspect that local PNG with its image-viewing tool. A saved
PNG is evidence of the viewport at that moment; it is not permission to click or
type into the desktop.

Stop the view bridge before restarting Blender:

```bash
blender-view stop
```

After Blender restarts, GNOME may require the Blender window to be selected and
shared again. Do not bypass the portal or request Remote Desktop input devices.

## Visual guidance overlay

The overlay is drawing-only and click-through. It supports normalized
window-relative rectangles, circles, lines, arrows, and labels. Before using it:

```bash
blender-overlay status
blender-overlay target
blender-overlay demo --ttl 10
```

Use the overlay only if the reported target frame and the temporary border
visibly match the Blender window. The XWayland fallback has previously anchored
annotations relative to the screen instead of the Blender window. If alignment
is wrong, immediately run:

```bash
blender-overlay clear
```

Then guide through Blender-native selection/highlighting and screenshots instead.
Never tell the user an annotation points to a control unless a screenshot or the
user has confirmed alignment.

For an aligned backend, a short-lived annotation can be sent as JSON:

```bash
blender-overlay show --json '{
  "ttl": 20,
  "annotations": [
    {
      "type": "arrow",
      "x1": 0.30,
      "y1": 0.20,
      "x2": 0.18,
      "y2": 0.12,
      "color": "#ffcc00ff",
      "width": 5
    },
    {
      "type": "label",
      "x": 0.31,
      "y": 0.20,
      "text": "Open this panel"
    }
  ]
}'
```

Prefer a finite `ttl`, and clear stale annotations when the viewport or active
workspace changes.

## Slider-crank-specific workflow

The linkage is intentionally a one-degree-of-freedom closed-loop mechanism, not
a freely moving two-link arm. Its Blender pose is solved directly from the
extended-assembly equations. Phobos link, joint, inertial, visual, and collision
metadata accompany that driven rig.

The coordinate convention is:

- motion in the XY plane;
- all hinge axes along Z;
- +Y from servo pivot A toward the ground; and
- `theta_deg` measured from +Y toward +X.

The saved defaults are:

| Parameter | Value |
| --- | ---: |
| Design-unit scale | 0.1 m/unit |
| L4 crank | 3.5 units / 0.35 m |
| L5 connector | 2.5 units / 0.25 m |
| L3 output leg | 3.0 units / 0.30 m |
| L3 mass | 1.0 kg |
| L4 mass | 0.30 kg |
| L5 mass | 0.25 kg |
| Default angle | +20 degrees |
| Geometric angle limit | approximately ±45.5847 degrees |
| Operational angle limit | approximately ±45.0847 degrees |

### Primary control

Use the specialized command so angle limits, dead-center intent, dependency
graph refresh, and verification happen together:

```bash
blender-control slider-crank-theta 30
```

Do not change the raw custom property through a generic property command. In
the Blender interface, the same control is
`SliderCrankController` → Object Properties → Custom Properties → `theta_deg`.

The exact toggle pose requires an explicit acknowledgement:

```bash
blender-control slider-crank-theta 0 --allow-dead-center
```

Crossing from a positive angle to a negative one also requires explicit intent:

```bash
blender-control slider-crank-theta -20 --allow-dead-center-crossing
```

The operational margin is the normal boundary. Reaching closer to the geometric
limit is exceptional and requires `--allow-geometric-limit`.

Physically, changing theta rotates L4 about fixed A. Fixed-length L5 closes the
loop at B, while B remains on the Y axis. L3 is rigid and translates with B, so
D moves linearly. At theta = 0 the mechanism is fully extended and the foot at D
meets the ground surface at y = 0.9 m. This is a dead-center/toggle condition;
the direct equations evaluate it deterministically, but a physical force model
is singular there.

### Validation

After every geometry, driver, mass, inertia, or collision change, run:

```bash
blender-control slider-crank-verify
```

Acceptance requires:

- command response `ok: true`;
- result `valid: true`;
- `physics_data_valid: true`;
- maximum point/length/closure error no greater than `1e-6` m;
- B's X and Z constraint errors within tolerance;
- positive principal inertias and midpoint COMs for L3, L4, and L5;
- the L3, L4, L5, foot, and ground collision objects; and
- gravity approximately `(0, +9.81, 0)` m/s².

A useful regression sweep is:

```bash
blender-control slider-crank-theta 0 --allow-dead-center
blender-control slider-crank-theta 45
blender-control slider-crank-theta -20 --allow-dead-center-crossing
blender-control slider-crank-theta 20 --allow-dead-center-crossing
blender-control slider-crank-verify
blender-control save
```

This deliberately checks full extension/contact, a near-limit retracted pose,
the mirrored angle branch, and the final safe pose. Cold-reopen the saved file
and run the validator once more after changing the builder or drivers.

### Rebuilding the generated scene

The builder enforces `l4 + l5 = 6` and `l4 > l5`:

```bash
blender-control slider-crank-build \
  --theta-deg 20 \
  --l4-units 3.5 \
  --l5-units 2.5 \
  --l3-units 3.0 \
  --unit-scale-m 0.1 \
  --m3-kg 1.0 \
  --m4-kg 0.30 \
  --m5-kg 0.25
```

When starting from another `.blend`, save the generated result under the
dedicated slider-crank filename rather than replacing the source model.

The direct drivers and unconstrained rigid-body simulation must not control the
same objects simultaneously. The current masses and collision shapes are Phobos
model data for downstream simulation and contact work; they do not by themselves
turn the driven Blender rig into a free dynamic simulation.

Also remember that URDF is a tree format. The Phobos hierarchy records the
useful link and joint metadata, but the B loop closure needs a simulator or
format that supports closed-chain constraints, or an explicit export-time
representation choice.

## How to teach while controlling Blender

For each short group of steps, use this pattern:

1. **Action:** say exactly what to click or what the agent will change.
2. **Purpose:** explain why that Blender/Phobos object or property exists.
3. **Physical consequence:** explain what would move, remain fixed, collide, or
   carry load in the real mechanism.
4. **Check:** tell the user what visual result to expect, then inspect it through
   the API or a screenshot.

Avoid long uninterrupted recipes. Wait for the user after two or three manual
actions, especially when selection order or mode matters. If a shortcut differs
on macOS, give the menu path first. When an operation makes an object disappear,
stop and inspect transforms, visibility, parent relations, mode, and scene before
adding further steps.

## Failure recovery

| Symptom | Response |
| --- | --- |
| `blender-control` socket missing | Check existing Blender processes and state JSON; start one intended file with `blender-phobos` |
| Commands reach the wrong file | Stop, query status, save any dirty work, close the extra process, then relaunch the intended file |
| Mutation returns an error | Read `rolled_back` and `backup`; query current state before retrying with changed arguments |
| Undo does not visibly restore a direct property | Query the property; the bridge brackets RNA writes with explicit before/after undo snapshots |
| Drivers do not move after a socket property write | Use the specialized slider-crank angle command; it tags and refreshes the dependency graph |
| Model looks right but verification fails | Trust coordinates and invariants over appearance; inspect the named failing error and COM/inertia data |
| Inertia or COM stays zero | Ensure the inertial collection remains dependency-graph evaluable; hiding an entire collection can suppress its drivers |
| Every link has the same viewport color | Inspect material slots; shared mesh datablocks can inherit an old slot 0 |
| View bridge waits for selection | Ask the user to choose Blender and click Share in the GNOME portal |
| Overlay appears near the screen corner | Clear it; verify window tracking with `target` and `demo`, or abandon overlay guidance |
| Blender must restart | Save, stop the view bridge, terminate normally, relaunch, poll status, verify, and re-share the Blender window |

## Security boundaries

The control bridge uses a same-user Unix socket, not TCP. Its runtime directory
is mode `0700`, the socket is mode `0600`, and Linux peer credentials must match
Blender's user ID. It exposes structured operations and an operator allowlist,
not arbitrary Python evaluation. Writes are restricted to the user's home
directory and `/tmp`.

The screenshot bridge requests only GNOME ScreenCast access and reports
`input_control: false`. The overlay is non-reactive and click-through. Do not
weaken these properties merely to make an interaction more convenient.

When a required operation is absent from the allowlist, first look for a safe
composition of existing structured commands. If a new capability is genuinely
needed, add the narrowest parameterized operation, retain backup/audit behavior,
compile-check the add-on, restart Blender, and test both success and rejection
paths.
