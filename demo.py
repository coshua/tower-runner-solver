import os
import torch
from datetime import datetime, timedelta
import time
from gym.wrappers import Monitor
from env import create_env
from agent_eval import AgentEval  # Or import your specific Agent class

start_time = time.time()
timestamp = datetime.now().strftime('%m%d%H%M')
video_folder = f"videos/{timestamp}/"
os.makedirs(video_folder, exist_ok=True)
# --- Set up parameters ---
class Args:
    seed = 123
    disable_cuda = False
    device = torch.device('cpu')
    environment_filename = './ObstacleTower/obstacletower'
    atoms = 51
    V_min = -10
    V_max = 10
    model = None
    worker = 45
    hidden_size = 512        # Add this
    noisy_std = 0.1          # Add this
    
args = Args()

# Enable CUDA if available and not disabled
if torch.cuda.is_available() and not args.disable_cuda:
    args.device = torch.device('cuda')

# --- Create the environment ---
env = create_env(
    args.environment_filename,
    custom=True,
    worker_id=47,  # Use an available worker id for demo
    realtime=False,
    device=args.device
)

# --- Wrap environment with Monitor to record video ---

env = Monitor(env, directory=video_folder, video_callable=lambda episode_id: True, force=True)

# --- Create the Agent and Load the Model ---
agent = AgentEval(args, env)

# Load your trained .pth model
checkpoint_path = 'results/worker36/intermediate_4345213_20250312_200830/model.pth'
state_dict = torch.load(checkpoint_path, map_location=args.device)
agent.online_net.load_state_dict(state_dict)
agent.online_net.eval()  # Set to evaluation mode

# --- Run a demo episode ---
state = env.reset()
done = False
total_reward = 0
step_count = 0
floor_count = 0

while not done:
    # Get action from the model (greedy means epsilon=0)
    action = agent.act_e_greedy(state, epsilon=0.0)
    
    # Step the environment
    state, reward, done, info = env.step(action)
    total_reward += reward
    step_count += 1
    
    if reward > 0:
        print(f"Step {step_count}: Got reward {reward:.2f} (Total {total_reward:.2f})")
        if reward >= 1.0:
            timestamp = time.time() - start_time
            elapsed_str = str(timedelta(seconds=int(timestamp)))
            floor_count += 1
            print(f"[{elapsed_str}]  🎉 Reached floor {floor_count}!")
    
    env.render()
print(f"[{elapsed_str}]")
print(f"Demo finished. Total reward: {total_reward}, Total steps: {step_count}")
print(f"Total floors reached (tracked): {floor_count}")
# --- Close environment (this flushes the video to disk) ---
env.close()

print(f"Demo video saved in the '{video_folder}' folder.")

