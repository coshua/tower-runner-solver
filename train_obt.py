import argparse
from datetime import datetime
import numpy as np
import torch
import os
from env import create_env, TimeoutMonitor
from agent import Agent
from memory import ReplayMemory
from test_obt import test
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--T-max', type=int, default=int(20e6), metavar='STEPS',
                    help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH',
                    help='Max episode length (0 to disable)')
parser.add_argument('--history-length', type=int, default=1, metavar='T',
                    help='Number of consecutive states processed')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ',
                    help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY',
                    help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.6, metavar='ω',
                    help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β',
                    help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=5, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(32e3), metavar='τ',
                    help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=64, metavar='SIZE', help='Batch size')
parser.add_argument('--learn-start', type=int, default=int(80e3), metavar='STEPS',
                    help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS',
                    help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N',
                    help='Number of evaluation episodes to average over')
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N',
                    help='Number of transitions to use for validating Q')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('environment_filename', default='./ObstacleTower/obstacletower', nargs='?')
#parser.add_argument('--docker-training', action='store_true')
parser.add_argument('--timeout-monitor', action='store_true')
parser.add_argument('--worker', type=int, default=0, help='Worker id for unique results folder')

#parser.set_defaults(docker_training=False)


# Simple ISO 8601 timestamped logger
def log(s):
    print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


def main():
    start_time = datetime.now()
    
    args = parser.parse_args()
    results_path = os.path.join('results', f'worker{args.worker}')
    os.makedirs(results_path, exist_ok=True)

    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(1, 10000))
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(np.random.randint(1, 10000))
        # Disable nondeterministic ops (not sure if critical but better safe than sorry)
        #torch.backends.cudnn.enabled = False
    else:
        args.device = torch.device('cpu')

    args.large = False
    args.skip_frames = 0
    args.random_aug = 0.

    # Environment
    train_env = create_env(
        args.environment_filename,
        worker_id=9,
        custom=True,
        large=args.large,
        skip_frames=args.skip_frames,
        random_aug=args.random_aug,
        #docker=args.docker_training,
        device=args.device
    )
    action_space = train_env.action_space

    test_env = create_env(
        args.environment_filename,
        custom=True,
        large=args.large,
        custom_reward=False,
        skip_frames=args.skip_frames,
        #docker=args.docker_training,
        device=args.device,
        worker_id=1,
    )

    mem = ReplayMemory(args, args.memory_capacity, obs_space=train_env.observation_space)
    val_mem = ReplayMemory(args, args.evaluation_size, obs_space=test_env.observation_space)

    # for debugging environment issues
    if args.timeout_monitor:
        train_env = TimeoutMonitor(train_env, mem)
        test_env = TimeoutMonitor(test_env, val_mem)

    # Agent
    dqn = Agent(args, train_env)

    priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
    time_step = 0
    done = True
    state = None
    while time_step < args.evaluation_size:
        if done:
            state = train_env.reset()
            done = False

        next_state, _, done, _ = train_env.step(action_space.sample())
        val_mem.append(state, None, None, done)
        state = next_state
        time_step += 1

    if args.evaluate:
        dqn.eval()  # Set DQN (online network) to evaluation mode
        avg_reward, avg_Q = test(args, 0, dqn, val_mem, evaluate=True)  # Test
        print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
    else:
        # Training loop
        dqn.train()
        done = True
        for time_step in tqdm(range(args.T_max), mininterval=60):
            if done:
                state = train_env.reset()
                done = False
                episode_reward = 0.0

            if time_step % args.replay_frequency == 0:
                dqn.reset_noise()

            action = dqn.act(state)
            next_state, reward, done, info = train_env.step(action)
            episode_reward += reward

            # Optional reward clip
            if args.reward_clip > 0:
                reward = max(min(reward, args.reward_clip), -args.reward_clip)
            mem.append(state, action, reward, done)

            if time_step >= args.learn_start:
                # Recompute priority weights, etc.
                ...
                # Train
                if time_step % args.replay_frequency == 0:
                    dqn.learn(mem)

                # Evaluate
                if time_step % args.evaluation_interval == 0:
                    dqn.eval()
                    avg_reward, avg_Q = test(args, time_step, dqn, val_mem, env=test_env)
                    print(f"[{datetime.now().isoformat()}] T = {time_step} / {args.T_max} | Avg. reward: {avg_reward:.2f} | Q: {avg_Q:.2f}")
                    #tqdm.write(f"[{datetime.now().isoformat()}] T = {time_step} / {args.T_max} | Avg. reward: {avg_reward:.2f} | Q: {avg_Q:.2f}")
                    dqn.train()

                # Update target net
                if time_step % args.target_update == 0:
                    dqn.update_target_net()
                    
            if (datetime.now() - start_time).total_seconds() >= 5 * 3600:
                checkpoint_path = os.path.join(results_path, f"intermediate_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{time_step}")
                os.makedirs(checkpoint_path, exist_ok=True)
                print(f"Saving intermediate model at step {time_step} to {checkpoint_path}")
                dqn.save(checkpoint_path)
                start_time = datetime.now() 
            state = next_state
            
    # After the training loop completes:
    final_model_path = f"results/final_model_worker{args.worker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dqn.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    train_env.close()


if __name__ == '__main__':
    main()
