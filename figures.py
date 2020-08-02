import argparse
import os
from pathlib import Path

import torch
from moviepy.editor import ImageSequenceClip

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
    )
    agent.policy_net.load_state_dict(torch.load(ckpt))
    agent.policy_net.eval()
    return agent, game


def _make_gif(agent, name, epsilon):
    fps = 15
    total_frames = 45 * fps
    frames = []
    for _ in range(total_frames):
        frames.append(agent.env.render(mode='rgb_array'))
        _, _, done = agent.step(epsilon)
        if done:
            agent.reset()
    g = ImageSequenceClip(frames, fps=15)
    g.write_gif(os.path.join('gifs', f'{name}.gif'), fps=fps)


def make_gifs(agent, game):
    os.makedirs('gifs', exist_ok=True)
    print('Make GIF of random play...')
    _make_gif(agent, f'{game}.random', 1.0)
    print('Make GIF of trained play...')
    _make_gif(agent, f'{game}.trained', 0.0)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--save_path', default='checkpoints')
    args = p.parse_args()

    for ckpt in Path(args.save_path).glob('*.pt'):
        agent, game = load_agent(ckpt)
        make_gifs(agent, game)
