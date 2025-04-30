```
import argparse
import os
from datetime import datetime
import numpy as np
import torch
from env import create_env
from agent import Agent

def run_episode(env, agent):
    state = env.reset()
    done = False
    episode_reward = 0.0
    while not done:
        # Use the agent’s act() method to select an action
        action = agent.act(state)
        state, reward, done, info = env.step(action)
        episode_reward += reward
    return episode_reward

def main():
    parser = argparse.ArgumentParser(description='Rainbow Evaluator')
    parser.add_argument('--model', type=str, required=True, default='./results/worker36/intermediate_4815006_20250313_010830/model.pth'
                        help='Path to the trained model state dict')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--num-episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--environment_filename', type=str, default='./ObstacleTower/obstacletower',
                        help='Path to Unity environment')
    parser.add_argument('--render', action='store_true',
                        help='Display screen during evaluation (if supported)')
    parser.add_argument('--worker', type=int, default=1,
                        help='Worker id for unique evaluation environment')
    args = parser.parse_args()

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(1, 10000))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    # Create the evaluation environment.
    # Note: Adjust the parameters (custom, large, custom_reward, etc.) as needed.
    eval_env = create_env(
        args.environment_filename,
        custom=True,
        large=False,
        custom_reward=False,
        skip_frames=0,
        device=args.device,
        worker_id=args.worker,
        # If your create_env supports a render flag, you may pass it here.
        render=args.render
    )

    # Instantiate the agent and load the trained model.
    agent = Agent(args, eval_env)
    agent.load(args.model)
    agent.eval()  # Set the agent to evaluation mode

    # Run evaluation episodes.
    rewards = []
    for episode in range(args.num_episodes):
        ep_reward = run_episode(eval_env, agent)
        rewards.append(ep_reward)
        print(f"Episode {episode + 1} return: {ep_reward}")

    avg_reward = np.mean(rewards)
    print(f"Average return over {args.num_episodes} episodes: {avg_reward}")

    eval_env.close()

if __name__ == '__main__':
    main()
```