# Running the JAX branch on Marshall

Branch: `jax-nelder-mead`. The parent legacy code revision is
`eb5bf3da41f684fc729788ccbd5c542136891454`. Use `git rev-parse HEAD` after pulling
to record the exact JAX revision; a branch name alone is not a run identifier.

## Current experiment

- Specification: MScal, seven structural parameters; unmarried scale fixed to 1.
- Primal objective only; proportional **entry and exit**, in training and plots.
- 600 inner Adam updates per trial, learning rate 0.001; 4,096 sampled states.
- Two hidden layers of 192 and 96 units (21,614 network parameters for MScal).
- Fresh male/female Dirichlet batches on each update; the advanced random key
  carries into the next Nelder–Mead trial.
- Annual beta 0.95; historical window 1999–2020, initialized from observed 2001.
- Historical KL and conditional diagnostics cover 2001–2020; the initial 2001
  contribution is zero. Inner Bellman draws still cover the 1999–2020 grid.
- Nelder–Mead outer search, simplex increment 0.3, maximum 1,000 iterations.
- Stop after 150 outer steps without a strict new global best; `--patience 0`
  disables this rule. Any finite improvement resets the counter, however small.
- The November behavior that retains the incumbent score after shrink remains.
- `--noload` starts coefficients at 0.1 with JAX random key 0. Backend random
  streams and network initialization are not identical to PyTorch.
- New checkpoint format: `hfdd/thetaMScal.npz` and `hfdd/xiMScal.eqx`.
- Progress history: `figures/training_history.csv`; save a copy after each run.
  Checkpoints are best evaluated pairs, not complete optimizer-resume states.

The larger-network experiment increases the hidden layers from 128/64 to
192/96 and inner updates from 300 to 600, retaining 4,096 draws per update.
The preceding configuration is recorded at commit
`d0f9c3fa89d9d72e4f3efbf44d21304d79bbc2d6`. Start this experiment with `--noload`;
older 128/64 `.eqx` checkpoints require a network template with their original
dimensions. Preserve the earlier run's checkpoints and history before restarting
if they are needed for comparison, since the output filenames are reused.
The wider network completed a CPU smoke check with all 600 updates and eight
synthetic states per update: loss and weights remained finite, weights stayed
FP32, and the inner solver printed nothing. The 4,096-draw GPU memory footprint
still needs to be checked on Marshall.

This changes the backend, network width, flow rule, and initialization from
the interrupted November PyTorch run. It is not a controlled backend-only
replication. See the [legacy comparison record](LEGACY_COMPARISON.md).

The precommit launch check fixed Streamlit device hashing in `load_data`: its
cache key identifies the JAX device by platform, device ID and process index.
This is a startup/cache fix, with a mocked-data regression test. No estimation
formula or optimizer setting was changed as part of that fix.

## Fetch and install

Run on Marshall with the existing project venv activated (Python 3.10 or 3.11):

```bash
cd ~/DynamicMatchingPy
git fetch origin
git switch jax-nelder-mead
git pull --ff-only

python -m pip install -r requirements.txt 'jax[cuda12]>=0.6.2,<0.7'
python -m pip check
```

On the first checkout, `git switch` creates a local branch tracking the unique
matching remote branch. Preserve local changes if Git reports a conflict;
these instructions do not discard server work.

Requirements use minimum versions rather than exact pins. Hub stays below 1
for compatibility with the existing Transformers 4.51.3 installation; the
project's loading calls do not require Hub 0.24.6. pandas stays below 3 because
preprocessing still uses `groupby(axis=1)`. JAX, Optax, and Equinox stay within
the release families used for this port; newer families can be evaluated
separately. Record the actual installed versions for each run with `pip freeze`
(included in the optional launch record below).

If the venv still contains Torch 2.5.1, it also needs a Torch upgrade: its
cuDNN 9.1 requirement conflicts with JAX's cuDNN >=9.8 requirement. Torch is
not a dependency of this JAX branch, so the command above does not upgrade it.
The [official Torch 2.8.0 CUDA 12.8 combination](https://pytorch.org/get-started/previous-versions/#v280)
(`torch` 2.8.0, `torchvision` 0.23.0, `torchaudio` 2.8.0) uses compatible cuDNN.

The repository requirements install CPU JAX by default. The CUDA extra enables
the NVIDIA GPU within the same JAX release family. CUDA 12
requires a sufficiently recent NVIDIA driver (Linux driver >=525). See
[JAX installation instructions](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu).
If local CUDA paths conflict with pip's bundled libraries, consult that page
before changing the environment.

Confirm the backend and dataset access:

```bash
python -c 'import jax; print(jax.__version__); print(jax.devices())'
python -c 'from datasets import load_dataset; d=load_dataset("StefanHubner/DivorceData", token=True); print(d["MS"].num_rows)'
test -d hfdd
```

`hfdd` should be the existing model-output directory or valid symlink. The
cached Hugging Face login/HF_TOKEN used by the legacy run is still required.
Training selects JAX's default device automatically: GPU when available,
otherwise CPU. No `JAX_PLATFORMS` setting is needed. The device check above
shows which backend is available before starting a long run.

## Launch

```bash
python main.py --spec MScal --train --noload
```

Nelder–Mead is always used for the outer parameters; `--neldermead` is retained
as a redundant compatibility flag. Adam remains the inner network optimizer.
The inner solver and individual trial evaluations print nothing. Each outer
step prints three coordinated lines:

1. Current total loss in cyan, global best total in green, and current inner
   residual in cyan, plus the number of outer steps since the last best.
2. Maximum absolute mass deviation in yellow, over both genders' type margins,
   all accepted sampled states, all 600 inner updates, and every trial evaluated
   in this outer step. This includes trials that were not selected as best.
3. Current parameters in cyan and global-best parameters in green, both shown
   to three decimal places. Rounding affects display only.

"Current" consistently means the best newly evaluated trial in this outer step.
The detailed conditional differences appear in grey only when that trial sets
a new global best. Inner loss still refers to its final training batch before
the last Adam update. The mass diagnostic observes each batch before its update;
it is not an additional validation pass and does not enter the objective.

`--patience 150` stops after 150 consecutive outer steps without a strict finite
improvement, saving the same best parameter/network pair as before. A best at
step 250 followed by no improvement therefore stops at step 400. This is a
practical stopping rule, not a convergence certificate: current trial losses
can rise while Nelder–Mead explores, whereas the saved best is nonincreasing.
`--patience 0` restores the full 1,000-step run. A step with no finite new trial
is displayed as `n/a` and cannot overwrite the best; an entirely nonfinite
first step stops with an error.

Historical KL cell contributions are computed in FP32 and summed in host FP64.
For MScal, this transfers just 480 FP32 values (1,920 bytes) across both genders
and 20 years. The total `KL + 2 * inner_loss` and the scores retained/returned by
Nelder–Mead preserve FP64. The inner residual itself remains FP32, as do neural
training, recursive predictions, and simplex coordinates. Global JAX x64 is
not enabled, so sampling and GPU matrix arithmetic retain their existing dtypes.
This improves numerical resolution of the outer score's accumulation; it does
not remove approximation error or noise already present in the FP32 terms.
A synthetic check found two predictions with tied old FP32 KL scores whose
FP64 scores differed by 1.86e-9; Nelder–Mead retained the lower FP64 score.
The compiled inner solve contained no FP64 operations and reproduced the
previous FP32 inner loss. GPU timing has not been benchmarked locally.

The quiet inner solver was checked on a small synthetic MScal problem on CPU:
the compiled 300-update solve returned exactly the same loss, network weights,
and random key as before, with no console output. Its maximum mass deviation
matched an independent observation of every update (5.25e-6 over the solve,
versus 1.79e-7 on the last update alone). Separate checks covered aggregation
across losing trials and reset between outer steps. The outer-loop checks
covered patience, a 1e-12 improvement resetting it, disabling it, nonfinite
trials, checkpoint retention, and the coordinated display. An earlier sampling
audit observed 1,200 distinct Dirichlet batch keys across two compiled solves.
These checks do not establish convergence of the server estimation.

## Optional background run and record

```bash
mkdir -p logs
run_id=$(date -u +%Y%m%dT%H%M%SZ)
git rev-parse HEAD > "logs/${run_id}_MScal.commit.txt"
git diff --no-ext-diff > "logs/${run_id}_MScal.worktree.patch"
python -m pip freeze > "logs/${run_id}_MScal.packages.txt"
python -c 'import jax; print(jax.__version__); print(jax.devices())' > "logs/${run_id}_MScal.devices.txt"
printf '%s\n' 'python -u main.py --spec MScal --train --noload' > "logs/${run_id}_MScal.command.txt"
nohup python -u main.py --spec MScal --train --noload > "logs/${run_id}_MScal.log" 2>&1 < /dev/null &
echo $! > "logs/${run_id}_MScal.pid"
tail -f "logs/${run_id}_MScal.log"
```

The first inner solve includes JIT compilation. `nohup` survives SSH logout,
not a server reboot. Do not run two estimations of the same specification in
this checkout concurrently: they share checkpoint and history filenames.
Local validation passed three existing checks, the focused data-cache check,
and a compiled synthetic 300-step inner solve on CPU. The updated unit tests
are kept locally and are not part of this branch update. These checks do not
establish convergence of a full estimation or GPU performance.
