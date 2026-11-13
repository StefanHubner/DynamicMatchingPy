# Legacy PyTorch checkpoint audit — 13 November 2026

The interrupted MScal run recovered a useful historical fit with the dual loss
disabled. Its saved parameters match the best row in the copied training history.
This establishes checkpoint integrity and a recovered fit; it does not establish
controlled optimizer convergence or explain the parameter differences by itself.
MScaltrend was not rerun.

Only this written record is versioned. Linked scripts, raw histories, plots,
and diagnostic artifacts under `output/` are retained locally and excluded
from Git; those links require the local audit output directory.

## Revisions and scope

| Item | Revision or location |
| --- | --- |
| October reference source | `b96c01ac732751e323e6df8779b70ce0d3c9ba40` |
| November legacy source | `eb5bf3da41f684fc729788ccbd5c542136891454` |
| October MScal checkpoint, `StefanHubner/DutchDivorce` | `67bf452c026f207b138b399eff70cbb819f9b771` |
| Equivalent October 27 MScal files | `eaa244363a7946b1160f183dc0bbbdd69f4f813f` |
| November 6 dual checkpoint | `8598e40cb5b60893e3f45f6bf95ce25887b5d899` |
| November 13 no-dual checkpoint | `91eee0e687da1b29ce6b6e06b7016615ae183d75` |
| Data used for replay, `StefanHubner/DivorceData` | `fedeb05309a7b2e0c7c729a6633f7b964fdd157e` |
| Copied server history | [training_history.csv](../output/diagnostics/2026-11-13_legacy_check/training_history.csv) |
| Frozen November Python modules and hashes | [source/](../output/diagnostics/2026-11-13_legacy_check/source/), [provenance.json](../output/diagnostics/2026-11-13_legacy_check/provenance.json) |

The checkpoint files are `thetaMScal.pt` and `xiMScal.pt`. The source commits
provide reference implementations; the October checkpoint does not contain an
exact training command, configuration, or source manifest. Its archived plotting
rule therefore cannot establish its training-time flow rule.

## What changed between October and November

Both reference implementations use annual beta 0.95, the same seven structural
parameters, unmarried scale fixed to one, calibrated treatment wedges, constrained
FOCs, and continuation derivatives. The
[frozen-source parity check](../output/diagnostics/2026-11-13_legacy_check/legacy_source_parity.json) confirms primal outputs,
losses, and gradients agree to floating-point precision when weights, parameters,
states, beta, and additive flows are matched. That check performs no training.

| Setting | October reference | November legacy run |
| --- | --- | --- |
| Flow configuration | Fixed entry and proportional exit by default; `--fixedflows` selects additive entry and exit | Additive entry and exit hardcoded |
| Inner Adam updates per trial | 300 | 600 |
| Dual loss | Absent | Explicitly disabled with `use_dual=False` |
| Network output count | 14 | 21; seven added outputs receive no direct primal-loss gradient |
| Nelder–Mead after shrink | Reevaluate all vertices | Keep incumbent score; evaluate the other vertices |
| Initialization | Actual run initialization and random seed unrecorded | First history row is consistent with a fresh all-0.1 parameter vector and the documented simplex construction; this is an inference |

Different initialization and random draws can change the finite inner solves.
The larger output layer also changes initialization; its extra outputs do not
change the primal mapping when corresponding weights are explicitly matched.
Individual welfare values from the inactive dual outputs are not trained by
this no-dual run.

Keeping an earlier incumbent score can matter because the inner network is
warm-started and training is stochastic. It is a plausible contributor to the
observed stalled search, but this audit does **not** demonstrate that the
optimizer change caused the October–November parameter differences.

The April-to-October corrections—per-economy population normalization,
constrained FOCs, regime timing, and coherent checkpoint saving—were already
present in the October reference. They cannot explain differences between these
October and November implementations. Comparisons with the older eight-parameter
checkpoint are a separate exercise; see [parameter metadata](../output/diagnostics/2026-11-13_legacy_check/parameter_comparison_metadata.json).

## History and checkpoint integrity

The history has 286 complete, finite rows out of a nominal 1,000 iterations. Its
best result is iteration 111:

`total 0.131007686 = historical KL 0.130631092 + 2 × inner loss 0.000188297`.

The saved theta matches that row exactly at float32 precision. There were no
further incumbent improvements for 175 iterations. Across the final 50 rows,
the largest parameter span is approximately `1.19e-6`, while total loss ranges
from 0.132852 to 0.136406. This supports a stalled parameter search with continuing
network and sampling variation, not a formal optimum or exact equilibrium
certificate. See [history summary](../output/diagnostics/2026-11-13_legacy_check/history_summary.json) and
[convergence plot](../output/diagnostics/2026-11-13_legacy_check/convergence_history.pdf).

The checkpoint preserves an evaluated theta/network pair. It does not preserve
the simplex, its scores, optimizer state, RNG state, epoch, or incumbent loss.
Restarting the command therefore starts a new search rather than resuming the
interrupted optimizer. The recorded inner loss precedes the final inner Adam
update; historical KL uses the network after that update.

## Fit and equilibrium under alternative flow rules

The following are saved-network replays, with no retraining. Historical KL uses
2001–2020; 2001 is the initialized observation.

| Checkpoint | Additive entry/exit | Fixed entry, proportional exit | Proportional entry/exit |
| --- | ---: | ---: | ---: |
| October | 0.136234 | 0.061899 | 0.071428 |
| November dual | 2.347888 | 2.311776 | 2.324820 |
| November no-dual | 0.130631 | 0.062322 | 0.068555 |

In particular, October's additive KL of 0.136234 is a replay result, **not** a
documented training fit. The archived October plots reproduce the mixed rule.
On the same 4,096 sampled states, normalized primal residuals are:

| Checkpoint | Additive entry/exit | Fixed entry, proportional exit |
| --- | ---: | ---: |
| October | 0.000804 | 0.000313 |
| November no-dual | 0.000191 | 0.001335 |

These residuals and the source defaults support different training flow rules
as an explanation, but do not prove October's actual configuration. Fixed and
mixed net flows agree at the calibration reference population and have different
population derivatives. Those derivatives enter continuation incentives, so
similar matching paths need not imply similar fitted surplus or scale parameters.

The policy network does not receive structural theta as an input. Theta affects
the inner equilibrium problem and welfare; frozen-network KL alone cannot
identify theta. Possible weak identification would require a moment or objective
profile under a common flow rule with the inner problem re-solved at each theta.
This audit does not supply that evidence.

Audit residuals use 100 projection iterations in evaluation mode; recorded
training loss uses 25 and different random draws. Residuals exclude invalid next
states and report their counts. Replaying another flow rule is not a newly solved
equilibrium. The new additive factual path becomes invalid in 2033; other replays
remaining valid through 2100 does not establish validity outside that horizon.

## Current JAX working tree

At the time of this audit, the uncommitted `jax-nelder-mead` working tree uses JAX,
Equinox and Optax, 300 inner steps, and `fixed=False` for proportional entry **and**
exit. Its Nelder–Mead implementation still retains the incumbent score after a
shrink. It uses new checkpoint formats and a different backend/initialization.
The PyTorch checkpoint comparison does not validate that migration's convergence.

## Reproduction and remaining provenance gaps

Run [check.py](../output/diagnostics/2026-11-13_legacy_check/check.py), [history_check.py](../output/diagnostics/2026-11-13_legacy_check/history_check.py),
[verify_legacy_parity.py](../output/diagnostics/2026-11-13_legacy_check/verify_legacy_parity.py), and
[plot_comparison.py](../output/diagnostics/2026-11-13_legacy_check/plot_comparison.py) from an environment with compatible
PyTorch, NumPy, pandas, datasets, and Matplotlib. This audit used
`/home/stefan/.venvs/main/bin/python`, with `MPLCONFIGDIR=/tmp/legacy-mpl`.
The scripts expect pinned checkpoint files in the repository's `.hfcache` and
the local Arrow dataset cache; see their paths before running elsewhere.
The main replay uses seed 20261113 for its common audit states. This audit seed
is not the seed of either original estimation run.

Results are in [comparison.json](../output/diagnostics/2026-11-13_legacy_check/comparison.json), [paths.npz](../output/diagnostics/2026-11-13_legacy_check/paths.npz),
[parameter comparison](../output/diagnostics/2026-11-13_legacy_check/parameter_comparison.csv), and
[comparison plots](../output/diagnostics/2026-11-13_legacy_check/legacy_comparison_plots.pdf). Existing checkpoint and figure
files were not overwritten.

The checkpoint pairs alone omit launch arguments, effective flow configuration,
data revision, code/dirty-tree manifest, package versions, training seed and RNG
state, initialization provenance, and optimizer state. The copied November
history resolves its best saved iteration, but not these missing October
provenance details. Future comparisons should save that configuration alongside
the weights and history so that differences in settings can be separated from
differences in optimization.

## Counterfactual comparison using the intended flow rule

`counterfactual_story.py` compares October and November saved networks using
proportional entry AND exit in both replays. No inner or outer optimization is
performed. CF1 retains the low-cost regime after 2008, whereas CFF switches it
off in 2009; both start from the observed 2001 allocation. CF0 omits the low-cost
regime throughout. The script checks finite/nonnegative allocations, population
accounting, and exact CF1/CFF agreement through 2008.

The CF1 effects in 2020 are similar despite the coefficient differences:

| Effect (counterfactual minus factual) | October | November |
| --- | ---: | ---: |
| Married adult share, percentage points | +2.745 | +2.880 |
| All partnered adult share, percentage points | +2.429 | +2.401 |
| Divorced unpartnered adult share, percentage points | +0.286 | +0.244 |
| Male-breadwinner share within married couples, percentage points | +2.598 | +3.155 |
| Annual aggregate flow utility, own-parameter units | +0.017081 | +0.016834 |

Both give more male-breadwinner and both-market-work couples and fewer
both-home-production couples through 2050. Annual flow welfare initially falls,
then becomes positive in 2014 (October) or 2015 (November). Discounted aggregate
flow-utility gains over 2009–2050 at beta=0.95 are positive in both (+0.229874 and
+0.213130, evaluated with each checkpoint's own preferences). These are finite
horizon sums, not certified infinite-horizon welfare. Truncating instead at 2020
changes the discounted-gain sign across estimates (+0.002917 versus -0.026356),
so a welfare claim must state its horizon. Cross-checkpoint welfare levels are
not directly comparable, and individual dual value heads are not used here.

The longer-run cohabitation/partnership story is less stable: in 2050 the
partnered adult share effect is +4.179 pp versus +2.459 pp, while marriage remains
+4.151 pp versus +3.598 pp. Divorce stocks are not annual divorce hazards; the
conditional unpartnered-probability effects need not share their sign.

Proportional flows prevent the checked negative-population failure but do not
establish a valid asymptotic forecast. Time is an input to the trained network;
the data end in 2020. In extrapolation to 2100, both annual welfare differences
turn negative in the 2080s, and November's marriage effect turns negative in
2095. Thus the shared qualitative story is supported over the displayed
2009–2050 window, not indefinitely.

Artifacts: [plots](../output/diagnostics/2026-11-13_legacy_check/counterfactual_story.pdf), [annual effects](../output/diagnostics/2026-11-13_legacy_check/counterfactual_story.csv),
[welfare horizon checks](../output/diagnostics/2026-11-13_legacy_check/counterfactual_story_summary.json),
[stock and specialization summary](../output/diagnostics/2026-11-13_legacy_check/cf_stock_summary.csv).
