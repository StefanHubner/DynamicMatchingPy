# Running the JAX branch on Marshall

Branch: `jax-nelder-mead`. The parent legacy code revision is
`eb5bf3da41f684fc729788ccbd5c542136891454`. Use `git rev-parse HEAD` after pulling
to record the exact JAX revision; a branch name alone is not a run identifier.

## Current experiment

- Specification: MScal, seven structural parameters; unmarried scale fixed to 1.
- Primal objective only; proportional **entry and exit**, in training and plots.
- 300 inner Adam updates per trial, learning rate 0.001; 4,096 sampled states.
- Annual beta 0.95; historical window 1999–2020, initialized from observed 2001.
- Nelder–Mead outer search, simplex increment 0.3, nominal 1,000 iterations.
- The November behavior that retains the incumbent score after shrink remains.
- `--noload` starts coefficients at 0.1 with JAX random key 0. Backend random
  streams and network initialization are not identical to PyTorch.
- New checkpoint format: `hfdd/thetaMScal.npz` and `hfdd/xiMScal.eqx`.
- Progress history: `figures/training_history.csv`; save a copy after each run.
  Checkpoints are best evaluated pairs, not complete optimizer-resume states.

This changes the backend, inner-step count, flow rule, and initialization from
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
