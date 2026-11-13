# DynamicMatchingPy

The primal model uses JAX, Equinox layers, Optax Adam for inner training, and
Nelder–Mead for outer estimation. Dependencies install CPU-only JAX.
See [Marshall setup and run commands](RUN_JAX.md) for the GPU install,
current experiment settings, and run-provenance recording.

The pre-dual residual construction and 300 inner steps are restored, retaining
the dated October corrections and graph cleanup. Entry and exit both use the
same `create_netflow(..., fixed=False)` proportional-flow setting.

`match_moments` returns its original seven outputs followed by the updated
network and random key. Checkpoints use NumPy `.npz` for theta and Equinox `.eqx`
for the network. Existing `.pt` checkpoints need conversion before loading;
`--noload` starts fresh.

## Legacy PyTorch checkpoint audit (2026-11-13)

The consolidated [run comparison](LEGACY_COMPARISON.md)
records the pinned revisions, verified results, configuration differences, and
limits of the convergence and parameter comparisons.

The legacy MScal run recovered the pre-dual quality of historical matching fit
with **`use_dual=False`**. The exact code commit is
`eb5bf3da41f684fc729788ccbd5c542136891454` (`no dual`, on `main`).
This validation concerns the committed PyTorch implementation, not the subsequent
JAX migration. The legacy run retained 600 inner Adam steps and additive entry
and exit; the current JAX entry point uses proportional entry and exit.

The saved model is Hugging Face repository `StefanHubner/DutchDivorce`, revision
`91eee0e687da1b29ce6b6e06b7016615ae183d75` (`legacy check`), files
`thetaMScal.pt` and `xiMScal.pt`. MScaltrend was not rerun.
The copied server history contains 286 completed iterations. Its best evaluated
pair, at iteration 111, matches the saved parameters exactly at float32 precision:
total loss 0.131007686 = historical KL 0.130631092 + 2 × inner loss 0.000188297.
There were no further improvements in 175 iterations, and parameter trials had
effectively stopped moving. This establishes a recovered fit and a stalled
search, not a proof of global optimality or exact equilibrium convergence.

Replaying the saved networks without retraining gives the following historical
KL values (2001–2020; lower is better):

| Checkpoint | Additive entry/exit | Fixed entry, proportional exit | Proportional entry/exit |
| --- | ---: | ---: | ---: |
| Pre-dual, October 22 | 0.136234 | 0.061899 | 0.071428 |
| Dual, November 6 | 2.347888 | 2.311776 | 2.324820 |
| Dual off, November 13 | 0.130631 | 0.062322 | 0.068555 |

The archived pre-dual plots used the earlier unintended mixed rule: fixed entry
with proportional exit. This was an implementation mistake, not an intended
model specification. The November 13 run itself estimated **fixed/fixed** flows
(additive entry and additive exit), not mixed flows. Therefore
the 0.13 versus 0.06 difference is a change in the population-flow rule, holding
the checkpoint fixed, not another estimation run. The intended current setting
is proportional **entry and exit** (`fixed=False`). Switching the flow rule in
this replay does not re-solve the equilibrium network under the new rule.
The new additive-flow factual path becomes invalid in 2033; its mixed and fully
proportional replays remain valid through the checked horizon of 2100.

Reproducible scripts, frozen legacy source, checkpoint provenance, parameter
comparisons, copied history, and fresh plots are in
[`output/diagnostics/2026-11-13_legacy_check`](../output/diagnostics/2026-11-13_legacy_check/).
The original checkpoints and existing figure files were not overwritten.
The `output/` artifacts are local only and are excluded from Git; the written
comparison record above is versioned.

Comparison scope: the October checkpoint's archived plotting rule does not
establish its training-time flow setting. In particular, its additive replay
also has KL about 0.13. The April-to-October fixes (per-economy population
normalization, constrained FOCs, regime timing, and coherent checkpoint saving)
were already present in October and cannot explain October-to-November
differences. A fresh frozen-source check in `legacy_source_parity.json` confirms
October/November primal output, loss, and gradient parity at matched inputs.
Remaining estimation differences include 300 versus 600 inner steps, extra
inactive dual output rows and different initialization, and retaining the old
Nelder–Mead incumbent score after shrink. The latter can matter with stochastic,
warm-started inner training. Similar saved-network fits alone do not establish
parameter identification: structural theta affects equilibrium training but
does not enter the saved policy network as an input.

The October default was mixed flows unless `--fixedflows` was supplied; November
hardcodes fixed/fixed. The earlier checkpoint is more consistent with the mixed
equations on the same 4,096 audit states (normalized primal residual 0.000313
mixed versus 0.000804 fixed); November shows the opposite (0.000191 fixed versus
0.001335 mixed). This supports, but does not prove without the October launch
configuration, different training flow rules. An additive replay KL near 0.13
does not identify the original training rule. The dataset repository's latest
revision remains `fedeb05309a7b2e0c7c729a6633f7b964fdd157e`, dated December 1, 2025;
there was no published October-to-November 2026 update to the flow inputs.
Fixed and mixed net flows coincide at the calibration reference population but
have different derivatives with respect to population, changing the Bellman
continuation incentives even when plotted population paths are similar.

The three-vintage parameter table/plot in the diagnostics directory uses the
pre-April checkpoint `ba7727a7e639833895967f9fa4241b385bbcde91` (saved March 24),
the October checkpoint `67bf452c026f207b138b399eff70cbb819f9b771`, and the November
checkpoint above. The first is the legacy eight-parameter model with three
free scales; the latter two have seven parameters and fix the unmarried scale
to one. Values are shown in original units, with scales exponentiated. The
April-era checkpoint should not be interpreted as having a verified training
command or an exact matching source/configuration snapshot.
