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

Run on Marshall (Python 3.10 or 3.11):

```bash
cd ~/DynamicMatchingPy
git fetch origin
git switch jax-nelder-mead
git pull --ff-only

python3 -m venv ~/.venvs/dynamicmatching-jax
source ~/.venvs/dynamicmatching-jax/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install 'jax[cuda12]==0.6.2'
python -m pip check
```

On the first checkout, `git switch` creates a local branch tracking the unique
matching remote branch. Preserve local changes if Git reports a conflict;
these instructions do not discard server work.

The repository requirements install CPU JAX by default. The additional pinned
CUDA installation enables the NVIDIA GPU without upgrading JAX itself. CUDA 12
requires a sufficiently recent NVIDIA driver (Linux driver >=525). See
[JAX installation instructions](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu).
If local CUDA paths conflict with pip's bundled libraries, consult that page
before changing the environment.

Confirm the backend and dataset access:

```bash
JAX_PLATFORMS=cuda python -c 'import jax; print(jax.__version__); print(jax.devices())'
python -c 'from datasets import load_dataset; d=load_dataset("StefanHubner/DivorceData", token=True); print(d["MS"].num_rows)'
test -d hfdd
```

`hfdd` should be the existing model-output directory or valid symlink. The
cached Hugging Face login/HF_TOKEN used by the legacy run is still required.
The CUDA-only backend selection prevents silent CPU fallback. CPU execution
can instead use `JAX_PLATFORMS=cpu` and omit the CUDA installation.

## Launch and keep a record

```bash
mkdir -p logs
run_id=$(date -u +%Y%m%dT%H%M%SZ)
git rev-parse HEAD > "logs/${run_id}_MScal.commit.txt"
git diff --no-ext-diff > "logs/${run_id}_MScal.worktree.patch"
python -m pip freeze > "logs/${run_id}_MScal.packages.txt"
JAX_PLATFORMS=cuda python -c 'import jax; print(jax.__version__); print(jax.devices())' > "logs/${run_id}_MScal.devices.txt"
printf '%s\n' 'JAX_PLATFORMS=cuda python -u main.py --spec MScal --train --noload --neldermead' > "logs/${run_id}_MScal.command.txt"
nohup env JAX_PLATFORMS=cuda python -u main.py --spec MScal --train --noload --neldermead > "logs/${run_id}_MScal.log" 2>&1 < /dev/null &
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
