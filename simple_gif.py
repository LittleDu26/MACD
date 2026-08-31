"""Generate a GIF of the supplied voxel robot under random actions."""

import argparse
import os

import gym
import imageio
import numpy as np

import evogym.envs  # noqa: F401 - registers the EvoGym environments
from evogym import get_full_connectivity


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Robot reconstructed from the supplied 5 x 5 image.
# 0: empty, 1: rigid, 2: soft, 3: horizontal actuator, 4: vertical actuator.
ROBOT_BODY = np.array(
    [
        [0, 3, 4, 1, 4],
        [2, 4, 4, 0, 4],
        [3, 4, 1, 2, 1],
        [2, 0, 1, 3, 4],
        [0, 0, 0, 0, 4],
    ],
    dtype=int,
)


def save_random_action_gif(
    output_path,
    env_name="Walker-v0",
    seed=20260830,
    steps=100,
    fps=20,
):
    """Run the robot with uniformly random actuator commands and save a GIF."""
    rng = np.random.default_rng(seed)
    env = gym.make(
        env_name,
        body=ROBOT_BODY,
        connections=get_full_connectivity(ROBOT_BODY),
    )
    env.seed(seed)
    env.reset()

    # Use a fixed camera so the full robot and ground remain visible.
    env.default_viewer.track_objects()
    env.default_viewer.set_pos((7.5, 3.5))
    env.default_viewer.set_view_size((15.0, 10.0))
    env.default_viewer.set_resolution((900, 600))

    frames = []
    done = False
    for _ in range(steps + 1):
        frames.append(env.render(mode="img"))
        if done:
            break

        action = rng.uniform(
            env.action_space.low,
            env.action_space.high,
            size=env.action_space.shape,
        ).astype(env.action_space.dtype)
        _, _, done, _ = env.step(action)

    env.close()

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, duration=1.0 / fps)

    print("Robot body:")
    print(ROBOT_BODY)
    print(f"Actuators: {env.action_space.shape[0]}")
    print(f"Random seed: {seed}")
    print(f"Frames: {len(frames)}")
    print(f"GIF saved to: {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", default="Walker-v0")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT_DIR, "visual", "random_actions_robot.gif"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_random_action_gif(
        output_path=args.output,
        env_name=args.env_name,
        seed=args.seed,
        steps=args.steps,
        fps=args.fps,
    )
