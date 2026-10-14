"""Record several checkpoints landing side by side on the same seeds (MP4 or GIF).

Every panel starts from the same initial state, so differences are due to the policy.

Usage:
    python -m scripts.record \
        --model "10k=runs/ppo_s0/checkpoints/step_10000.zip" \
        --model "100k=runs/ppo_s0/checkpoints/step_100000.zip" \
        --model "final=runs/ppo_s0" --seeds 0 1 2 --out videos/ppo_progress.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lunar.envs import make_env
from lunar.evaluation import classify_outcome
from lunar.training import load_model

PANEL_SIZE = (360, 240)
FPS = 50  # LunarLander's native render fps
HOLD_FRAMES = 40  # pause on the final frame so the outcome can be read


def rollout_frames(model_path: Path, seed: int) -> tuple[list[np.ndarray], str]:
    """Play one deterministic episode and return its frames and a result caption."""
    model = load_model(model_path)
    env = make_env(render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames, total = [env.render()], 0.0
    while True:
        action = int(model.predict(obs, deterministic=True)[0])
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        frames.append(env.render())
        if terminated or truncated:
            outcome = classify_outcome(terminated, truncated, float(reward))
            env.close()
            return frames, f"{outcome.value.upper()}  {total:.0f}"


def annotate(frame: np.ndarray, title: str, caption: str, font: ImageFont.ImageFont) -> np.ndarray:
    image = Image.fromarray(frame).resize(PANEL_SIZE, Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(image)
    draw.text((8, 6), title, fill=(255, 255, 255), font=font)
    if caption:
        draw.text((8, PANEL_SIZE[1] - 24), caption, fill=(255, 230, 120), font=font)
    return np.asarray(image)


def compose_episode(
    panels: list[tuple[str, list[np.ndarray], str]], font: ImageFont.ImageFont
) -> list[np.ndarray]:
    """Tile the panels horizontally; finished episodes freeze on their last frame."""
    length = max(len(frames) for _, frames, _ in panels) + HOLD_FRAMES
    out = []
    for t in range(length):
        tiles = []
        for title, frames, result in panels:
            done = t >= len(frames) - 1
            tiles.append(
                annotate(frames[min(t, len(frames) - 1)], title, result if done else "", font)
            )
        out.append(np.concatenate(tiles, axis=1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help='"label=path"')
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--out", type=Path, required=True, help=".mp4 or .gif")
    args = parser.parse_args()

    font = ImageFont.load_default(size=16)
    models = [(label, Path(path)) for label, path in (s.split("=", 1) for s in args.model)]
    frames: list[np.ndarray] = []
    for seed in args.seeds:
        panels = []
        for label, path in models:
            episode, result = rollout_frames(path, seed)
            panels.append((f"{label}  seed {seed}", episode, result))
            print(f"seed {seed} {label}: {result}")
        frames.extend(compose_episode(panels, font))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".gif":
        imageio.mimsave(args.out, frames[::2], duration=2 / FPS, loop=0)
    else:
        imageio.mimsave(args.out, frames, fps=FPS, macro_block_size=8)
    print(f"saved {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
