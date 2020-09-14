import argparse
import os
import time
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from moviepy.editor import ImageSequenceClip
from tqdm import trange

from train import Agent, parse_args


def load_agent(ckpt):
    game = os.path.basename(ckpt)  # checkpoints/Pong-v0.pt -> Pong-v0.pt
    game, _ = os.path.splitext(game)  # Pong-v0.pt -> Pong-v0
    args = parse_args([])
    agent = Agent(
        game=game,
        replay_buffer_capacity=args.replay_buffer_capacity,
        # Minimum possible for agent.__init__()
        replay_start_size=args.batch_size,
        batch_size=args.batch_size,
        discount_factor=args.discount_factor,
        lr=args.lr,
        print_self=False,
    )
    agent.policy_net.load_state_dict(torch.load(ckpt))
    agent.policy_net.eval()
    return agent, game


def _make_gif(agent, name, epsilon, n_seconds):
    fps = 15
    total_frames = n_seconds * fps
    frames = []
    for _ in range(total_frames):
        frames.append(agent.env.render(mode='rgb_array'))
        agent.step(epsilon)
        # the agent resets itself and the environment when done
    g = ImageSequenceClip(frames, fps=fps)
    g.write_gif(os.path.join('gifs', f'{name}.gif'), fps=fps)


def make_gifs(agent, game, n_seconds):
    print(f'Make GIFs of {game}...')
    os.makedirs('gifs', exist_ok=True)
    _make_gif(agent, f'{game}.random', 1.0, n_seconds)
    _make_gif(agent, f'{game}.trained', 0.0, n_seconds)


def _run_game(agent, epsilon):
    episode_reward = 0.
    while True:
        reward, _, done = agent.step(epsilon, clip_reward=False)
        episode_reward += reward
        if done:
            # the agent resets itself and the environment when done
            break
    return episode_reward


def make_scores(agent, game, n_episodes):
    print(f'Make scores of {game}...')
    trained_rewards = []

    for _ in trange(n_episodes, desc='Trained episodes'):
        trained_rewards.append(_run_game(agent, 0.))

    random_rewards = []
    for _ in trange(n_episodes, desc='Random episodes'):
        random_rewards.append(_run_game(agent, 1.))

    return (np.mean(trained_rewards), np.std(trained_rewards),
            np.mean(random_rewards), np.std(random_rewards))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--save_path', default='checkpoints')
    p.add_argument('--n_episodes', default=10, type=int, help='# episodes for measuring scores')
    p.add_argument('--n_seconds', default=10, type=int, help='GIF duration')
    args = p.parse_args()

    scores = {}
    for ckpt in Path(args.save_path).glob('*.pt'):
        agent, game = load_agent(ckpt)

        # GIFs
        make_gifs(agent, game, args.n_seconds)

        # Scores
        t_mean, t_std, r_mean, r_std = make_scores(agent, game, args.n_episodes)
        scores[game] = {
                'trained_mean': t_mean,
                'trained_std': t_std,
                'random_mean': r_mean,
                'random_std': r_std}

        pprint(scores)
