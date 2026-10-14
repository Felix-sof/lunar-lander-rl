# LunarLander RL warm-up

Teaching an agent to land a rocket softly with reinforcement learning (Gymnasium
`LunarLander-v3`, Stable-Baselines3): PPO vs DQN, and what happens when you change the
reward. 15 training runs, 3 seeds per setting, every number evaluated over 100 episodes.
The full lab notebook with all numbers and lessons learned is in [EXPERIMENTS.md](EXPERIMENTS.md).

![The same PPO agent after 10k, 50k, 250k, 500k and 1M training steps](media/ppo_progress.gif)

*One PPO agent at five points in training, same starting state. 10k: untrained, flies off.
50k: has learned to hover and never lands. 250k: tries to descend and crashes. 500k: first
landings. 1M: lands every time (100/100 evaluation episodes).*

![Summary of all agents](figures/summary.png)

## Findings

### 1. PPO solves it on every seed, but learning is not a straight line

All three PPO seeds pass the "solved" threshold of 200 (262.7 / 243.3 / 258.8 over 100
episodes). On the way, the agent first learns to **hover** (a safe local optimum: no crash
penalty) and gets *worse* for a while before it learns to land. Looking at the return
curve alone, 50k steps (-8) looks better than 250k (-149). Watching the agent shows that
50k is a dead end.

![PPO training curves](figures/ppo_baseline.png)

### 2. DQN learns faster, then forgets

![PPO vs DQN landings](media/ppo_vs_dqn.gif)

*Left: PPO at 1M steps. Middle: DQN at its best checkpoint (500k). Right: the same DQN run
at 1M steps.*

DQN (off-policy, replay buffer) is far more sample-efficient: at 250k steps its seeds score
+189 to +220 while PPO is still at -149. But all three DQN seeds **collapsed** later (one
reached -1079 at 500k, another ended up never firing its main engine). The final DQN models
average -196, no better than random. PPO's clipped on-policy updates never lost what they
had learned. PPO also needed only 7-15 min per 1M steps on CPU, while DQN took ~2 h on GPU.

![PPO vs DQN curves](figures/ppo_vs_dqn.png)

### 3. The agent optimizes what you reward, not what you mean

![Fuel penalty 0x, 1x, 3x, 10x](media/fuel_shaping.gif)

*Same state, four agents trained with different fuel prices. Free fuel: slow, still
hovering. 1x and 3x: clean landings. 10x: crashes on purpose.*

| fuel penalty | mean return (original reward) | behaviour |
|---|---|---|
| 0x (free) | 206.2 | lands slowly, ~1.8x main-engine use, up to 10% of episodes hover until timeout |
| 1x (original) | 254.9 | efficient landings |
| **3x** | **267.9** | best agent *and* fastest learner |
| 10x | -33.0 | **crashes deliberately** in 100% of episodes: a crash (-100) is cheaper than the fuel for a landing (~-360) |

All variants are scored with the *original* reward. Shaping the reward in the right
direction (3x) beat the original reward on its own metric. Pushing the same knob too far
(10x) made the agent choose to crash.

![Fuel shaping curves](figures/fuel_shaping.png)

Full-length videos (3 starting states each): [ppo_progress.mp4](media/ppo_progress.mp4),
[ppo_vs_dqn.mp4](media/ppo_vs_dqn.mp4), [fuel_shaping.mp4](media/fuel_shaping.mp4).

## Setup (Windows, Python 3.11)

Tested with Python 3.11, gymnasium 1.0.0, stable-baselines3 2.8.0, torch 2.6.0+cu124.

### 1. Protect the CUDA build of torch

pip sometimes replaces a CUDA torch build with the CPU wheel when resolving another
package's dependencies. Every install below uses `constraints.txt`, which pins the exact
torch build, so pip fails loudly instead of swapping it. Check after installing:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 2. Box2D on Windows

`gymnasium[box2d]` depends on `box2d-py==2.3.5`, which has no Windows wheel and is compiled
from source. It needs SWIG and the MSVC C++ build tools (Visual Studio "Desktop development
with C++" workload). A plain `pip install "gymnasium[box2d]"` fails in two ways:

1. `error: command 'swig.exe' failed: None`: swig is not installed yet. pip installs
   everything in one transaction, so swig from the same command is not available while
   box2d-py builds.
2. `ModuleNotFoundError: No module named 'swig'` from `swig.exe`: pip builds in an isolated
   environment, where the `swig.exe` launcher cannot find the `swig` Python package.

Working sequence:

```powershell
python -m pip install -c constraints.txt swig
python -m pip install -c constraints.txt --no-build-isolation box2d-py==2.3.5
python -m pip install -c constraints.txt "gymnasium[box2d]==1.0.0" stable-baselines3 matplotlib "imageio[ffmpeg]"
python -m pip install -c constraints.txt pytest black ruff
```

Verify:

```powershell
python -c "import gymnasium as gym; gym.make('LunarLander-v3').reset(seed=0); print('ok')"
```

## Project layout

```
lunar/          library code
  envs.py         env factories (Monitor sits *inside* reward shaping, so logs show the original return)
  wrappers.py     FuelCostWrapper: rescales the fuel penalty for reward-shaping experiments
  evaluation.py   episode rollouts that also record the outcome (landed / crashed / timeout) and fuel use
  callbacks.py    periodic evaluation to eval.csv, milestone + rolling checkpoints
  training.py     hyperparameters, train / resume, model loading
  power.py        blocks Windows idle sleep during training
scripts/        command-line entry points (run with python -m scripts.<name>)
experiments/    job files and a PowerShell launcher for the experiment runs
tests/          pytest suite
figures/        generated plots (committed)
results/        100-episode evaluation summaries as JSON (committed)
media/          GIFs and MP4s used in this README (committed)
runs/, videos/  training artifacts (git-ignored)
```

## Usage

All commands run from the project root.

```powershell
# Random-agent baseline (100 episodes)
python -m scripts.random_baseline

# Train. --timesteps is the absolute total; --resume continues from checkpoints/latest.zip
python -m scripts.train --algo ppo --seed 0 --timesteps 1000000
python -m scripts.train --algo dqn --seed 0 --timesteps 1000000 --device cuda
python -m scripts.train --algo ppo --fuel-main-scale 0 --fuel-side-scale 0 --run-name ppo_fuel0_s0
python -m scripts.train --algo ppo --seed 0 --timesteps 1500000 --resume

# Run a whole experiment (one job per line, logs in runs/logs/)
powershell -File experiments/run_parallel.ps1 experiments/ppo.txt -MaxParallel 1

# Evaluate over 100 episodes (original reward)
python -m scripts.evaluate runs/ppo_s0 runs/dqn_s0

# Plot curves (seed mean + min-max band)
python -m scripts.plot --group "PPO=runs/ppo_s*" --group "DQN=runs/dqn_s*" --out figures/ppo_vs_dqn.png --wall-time

# Summary bar chart from results/*.json
python -m scripts.plot_summary --out figures/summary.png

# Watch live
python -m scripts.watch runs/ppo_s0
python -m scripts.watch runs/ppo_s0/checkpoints/step_100000.zip --episodes 3

# Side-by-side video of checkpoints on the same seeds
python -m scripts.record --model "10k=runs/ppo_s0/checkpoints/step_10000.zip" --model "final=runs/ppo_s0" --seeds 0 1 2 --out videos/ppo_progress.mp4

# Tests and formatting
python -m pytest
python -m black lunar scripts tests
python -m ruff check lunar scripts tests
```

Each run directory contains `config.json`, `eval.csv` (one row per evaluation:
return, landed/crashed/timeout rates, fuel use, wall time), `sb3_log/progress.csv`
(SB3 internals such as entropy and value loss) and `checkpoints/`
(`step_<N>.zip` milestones, `latest.zip` for resuming, `final.zip`).

## Windows pitfalls hit in this project

- **SubprocVecEnv + TensorFlow installed:** on Windows every env worker is a fresh
  `spawn`ed Python process that re-imports SB3, and SB3's logger pulls in TensorFlow via
  tensorboard (~320 MB, ~4 s per process). Starting 48 workers at once ran out of RAM; some
  workers died mid-import (`partially initialized module 'pandas'`), and their parent
  processes **hung forever** waiting on the dead workers instead of exiting.
  `lunar/__init__.py` marks TensorFlow as unavailable for this project's processes, and
  PPO runs (16 workers each) are launched one at a time.
- **Sleep:** `lunar/power.py` holds `ES_SYSTEM_REQUIRED` during training. This blocks idle
  sleep only; closing the laptop lid still suspends the machine.
- **Non-ASCII path (`Masaüstü`):** TensorBoard is not used; logs are plain CSV. Console
  output is ASCII-only.
- **Resume semantics:** SB3's `learn(total_timesteps=n, reset_num_timesteps=False)` trains
  `n` *more* steps. `scripts/train.py` treats `--timesteps` as the absolute target and passes
  the difference (tested in `tests/test_training.py`).
- **PPO step counts round up** to a multiple of `n_steps * n_envs` (1024 x 16 = 16,384), so a
  "1M step" PPO run actually trains 1,015,808 steps.
