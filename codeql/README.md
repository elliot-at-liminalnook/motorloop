<!-- SPDX-License-Identifier: MIT -->
# CodeQL audit queries

Custom queries against the Python stack for defect classes this project has
actually been bitten by. Run them with the CodeQL CLI (installed at
`~/.local/opt/codeql`, on PATH as `codeql`):

```bash
# one-time (and after large refactors): build the Python database
LGTM_INDEX_FILTERS=$'exclude:**/.venv-warp\nexclude:**/out\nexclude:**/build\nexclude:**/site-src' \
  codeql database create out/codeql-db --language=python --source-root . --overwrite

# run every audit query
codeql database analyze out/codeql-db codeql/ \
  --format=sarif-latest --output=out/codeql-results.sarif
# human-readable instead:
codeql database analyze out/codeql-db codeql/ --format=csv --output=/dev/stdout
```

| query | defect class it hunts |
| --- | --- |
| `DiscardedParseKnownArgs.ql` | launcher flags silently swallowed by a lenient CLI adapter (the PBT dead-search-space bug) |
| `ArgparseDestNeverRead.ql` | options parsed but never consumed — dead CLI surface that lies to callers (found a --budget cap that capped nothing) |
| `DeadModuleConstant.ql` | ALL_CAPS constants no code loads (the sprint/hop reward-graveyard pattern; found 45 on first run) |
| `GeneratorlessTorchRandom.ql` | global-RNG draws in env code that break per-world replay (the trainer is deliberately exempt: its global RNG is checkpointed) |
| `HostSyncInStepPath.ql` | .item()/.numpy() in per-step env methods — GPU→host syncs that defeat the captured-graph one-stream discipline |
| `TorchLoadUnsafeWeightsOnly.ql` | checkpoint loads that unpickle arbitrary code (Joern J4) |
| `CheckpointLoadWithoutContract.ql` | shape-only checkpoint loads that reinterpret conditioning channels (Joern J1: the v1→v2 bypass) |
| `NonRecursiveSourceGlob.ql` | provenance fingerprints that miss nested packages (Joern J5) |
| `OpponentNeverReset.ql` | stored recurrent opponents whose hidden state crosses episode boundaries (Joern J2) |

Reviewed-and-kept findings live in [`ACCEPTED.md`](ACCEPTED.md); a rescan is
clean when its only hits are that baseline.

The standard generalized sweep is
`codeql database analyze out/codeql-db codeql/python-queries:codeql-suites/python-security-and-quality.qls ...`;
triage its `py/empty-except`, `py/file-not-closed`, and `py/init-calls-subclass`
findings with extra suspicion — those are the classes this codebase actually
exhibits. `py/call-to-non-callable` on `policy(obs)` call sites is a known
false positive (closure-defined `LoadedPolicy.__call__` defeats points-to).

Add new queries as `.ql` files here; keep each one tied to a defect class the
repo has really exhibited, with the incident named in the docstring.
