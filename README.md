## Video
[![TowerRunners - Obstacle Tower Final Demo](https://img.youtube.com/vi/ZI62ny0AzuY/0.jpg)](https://www.youtube.com/watch?v=ZI62ny0AzuY)

### [Full Video](https://www.youtube.com/watch?v=ZI62ny0AzuY)

## Project Summary
Our team, Tower Runners, worked on developing a reinforcement learning agent to navigate through the Obstacle Towers environment. Obstacle Towers was a game built for an AI and machine learning competition to benchmark the capabilities of the participant's agents. These capabilities include being able to solve intricate puzzles, path finding in an environment with few yet sparse rewards, computer vision, and determining how to balance exploration and exploitation. Within the Obstacle Tower itself, the agent is tasked with the overarching goal of ascending to the highest floor possible within a predetermined amount of time. This task is comprised of navigating through the rooms, collecting keys or solving puzzles to unlock doors (when necessary), and ascending to higher floors, where difficulty increases as the agent reaches higher floors.

The difficulty of the task primarily lies within the nature of sparse rewards and the agent's ability to make generalizations in a randomly generated environment. In reinforcement learning, agents are rewarded for behavior that lead to ideal states and outcomes, and are penalized for the opposite. It's crucial for an RL agent to receive the necessary feedback to determine what's acceptable behavior in order for it to complete its task while simultaneously maximizing rewards. As previously mentioned, an agent in Obstacle Tower is operating under a sparse reward environment and it's not 100% clear the actions it should take to navigate to a door because there may be moving obstacles in its way. It will have a difficult time assessing when a door requires a key to be unlocked (on top of identifying the key), and figuring how to solve a puzzle as well.

## Approaches

### PPO and Randomized Action (baseline approaches)
A baseline model that used to benchmark the other methods we planned to use was by using a Random Policy to sample moves to our agent. This model omitted the need for training because it would continuously decide on random actions to take until the end of each episode. Getting started with our agent, we wanted to examine the action space of the agent in Obstacle Towers. After watching several episodes of the agent in action, we took note of the different actions it could choose from and which included 54 discrete actions:
* Move forward, backward, no move
* Move left, right, no move
* Turn left, right, no turn
* Jump, no jump

As one would presume, the agent was acting unintelligibly, jumping and moving in all directions while turning on some occasion. This resulted in the agent on getting stuck in sections of the starting room without making progress towards any of the doors.

Another method we started out with first was training our reinforcement learning model using the Proximal Policy Optimization algorithm provided by the stable-baselines3 library. This model served as our baseline approach because it was easy to implement to start figuring how the agent operated in the Obstacle Tower environment. We used the Multi-layer Perceptron (MLP) and Convolutional Neural Network (CNN) policies. The former was used to determine how well the agent could perform out of the box compare to the randomized approach, but the latter policy felt the most appropriate because the observation was a 84x84 RGB image. Using a CNN policy meant allow spatial features to be extracted, allowing the agent to recognize the important objects such as the keys, doors, and terrain that it normally bumps into. We also decided to use PPO because it's an on-policy algorithm which means it learns using the most recent experiences and updates its current policy accordingly. This ensures the model isn't learning from previous encouters that are less relevant considering the placements and types of puzzles changes as it progress through higher floors. This way, we felt PPO would provide our agent with some level of adaptability to newer environments.

### Improving Upon PPO With Reward Shaping
A method we used to train our reinforcement learning model was using the Proximal Policy Policy algorithm with reward shaping to guide the agent to completing objectives e.g. collecting keys, solving puzzles, pathing through the doors, and ascending to higher floors. In the early stages of reward shaping, we implemented a straightforward way to let the agent know the type of actions we wanted to see more of by rewarding it when it would ascend to higher floors and the count for the keys collected increased. After several sessions of training the agent with this simple process of rewarding the agent, we noticed a major issue where the agent was having a difficult time finding pathing to the doors and exploring the rooms in a manner that would help them collect the keys. Therefore, it meant there were only a few occasions over several episodes of training where the agent would accomplish these tasks. Below are the hyperparameters that was used in both instances where we trained with the PPO model. Over several sessions of training where we adjusted the different hyperparameters, we found this was the most stable in yielding higher average rewards. Additionally, the models were trained for 2 million timesteps.

![PPO-Hyperparameters](./static/PPO_hyperparameter.png)

### Ideas of Incorporating Computer Vision
The idea came from most of the map is quite dull, except the doors which are bright colors, whether yellow, green, or red. The observation space is 84 x 84 x 3, with the 3 being color as it's in RGB, so the plan was to train the model using computer vision to help it go towards the vibrant doors. Initially, we hoped the model would the computer vision to help make better decisions. We were planning to implement ResNet, a CNN from torch, however these plans ended up backfiring as it took way too long to train the model and was completely unfeasible with what we were trying to achieve.

### Rainbow DQN
We moved to use the Rainbow Deep Q-Network (Rainbow DQN). Rainbow DQN integrates six major improvements over the standard DQN algorithm, each addressing key limitations commonly encountered in complex reinforcement learning tasks. Specifically, Double DQN reduces the overestimation bias in value predictions, leading to more stable learning. Prioritized Experience Replay enables the agent to focus on rare and valuable experiences, such as collecting keys and reaching new floors, which is crucial in sparse reward settings. The Dueling Network Architecture helps the agent differentiate between the value of being in a particular state (e.g., standing near a door) and the advantage of taking specific actions from that state. Multi-step learning accelerates reward propagation across time steps, allowing the agent to more effectively associate delayed rewards with earlier actions. Distributional RL models the entire distribution of possible future returns rather than a single expected value, improving the agent’s ability to manage the uncertainty inherent in Obstacle Tower’s randomized layouts. Finally, Noisy Networks introduce parameterized exploration directly into the network, enabling more efficient and directed exploration without relying on ε-greedy strategies. Together, these components make Rainbow DQN well-suited for mastering the Obstacle Tower, as it balances exploration and exploitation while efficiently learning from sparse and delayed feedback in a highly dynamic environment.



## Evaluation
We are able to run and test different algorithms in our model to see how they perform, so the environment does run correctly, albeit takes an immensely large amount of time to run. Our evaluation consists of quantitative and qualitative metrics:
### Quantitative Metrics
For quantitative metrics, we evaluated our project observing the ep_rew_mean collected in our training, and the average floors climbed and rewards for testing the classifiers. In the training phase, the rewards, length, and time spent per iteration are saved into a CSV file, and you can observe that as time goes on and more iterations are completed, each of the different RL algorithms becomes more successful by displaying higher rewards, obviously at different rates. Another observation we have made is that when the reward increases, so does the length of the episode. This is because going to the next floor gives you additional time, hence increasing the length of the episode.

Average Episode Reward: Measures learning improvement over time.

Episode Length: Tracks how efficiently the agent solves a level.

Generalization Performance: Tested by training on seeds/towers and evaluating on unseen ones.

In order to measure the performance of our RL agent using the Proximal Policy Optimization model, we ran several training sessions and recorded the average reward that it accumulated over time. Below, we've included the plot of one model that used the Multi-layer Perception policy, and the second model that used a residual network (Convolution Neural Network) policy. Before performing analysis of the models that we trained, we hoped the average reward accumulated by the agent using the residual network would've outperformed the agent that didn't. The original model that's barebones using only PPO obtained higher mean rewards and it also increased throughout it's training process. So the next thing we have to explore is how to fine-tune this portion of our model so the agent can fully utilize feature detection of the symbols and colors on the doors of various levels. Overall, we believe there's more to be understood about the affects of the hyperparameters e.g. learning rate, gamma (far-sightedness), batch size, and so on because our model was having trouble making progress by moving through the doors. We also need to increase the number of timesteps to train the models even longer.

![MLP and CNN Policy Models](./static/tower_with_cnn.png)

After setting up PPO with rewards, although slow, we did see a great jump in the average episode reward. Instead of remaining below 1, signifying the agent was often stuck trying to get up to the 1st floor from floor 0, the agent was now averaging well above 1, and for a short period of time around 1.5 million timesteps, it was averaging above 2. This shows that the agent was consistently getting to the 2nd floor, and gained even more rewards after that. This means that the agent was gaining rewards from the 2nd floor completing puzzles, or it was occasionally make it to the 3rd floor as well.

![PPO model rewards](./static/PPO_average_rewards.png)
<img width="631" alt="image" src="https://github.com/user-attachments/assets/0a64e36c-a5a3-48e4-abf8-b2e3317945f4" />

**Rainbow DQN**


We logged average episode rewards and Q-values at regular intervals (every 100,000 timesteps), and periodically saved intermediate models (every 6 hours) for further evaluation and demonstration.

The training was conducted for 5 million timesteps. Models were evaluated both quantitatively (by measuring average reward and Q-values) and qualitatively (through visual inspection of recorded agent runs). The training environment used sparse rewards, and the agent received feedback primarily upon clearing floors and completing key tasks.

![Reward Chart](./static/Reward_chart.png)
<img width="752" alt="image" src="https://github.com/user-attachments/assets/1fbe5c7a-0201-4b91-899f-54cc9bd4c09c" />

![Q value Chart](./static/Q_chart.png)
<img width="758" alt="image" src="https://github.com/user-attachments/assets/2a84bc62-7196-4efa-845e-0a0094a9bc49" />


Initially, the agent struggled to make progress, with average rewards remaining close to zero for the first 1.5 million timesteps. This was expected due to the sparse reward structure of the environment. However, a notable breakthrough occurred at around 1.5 million timesteps, where the agent's average reward began to increase, reaching 0.80 at 1.7M and 1.30 at 2.0M timesteps. This suggests that the agent successfully explored its environment and learned effective policies for clearing early floors.

From 3 million timesteps onward, we observed a steady climb in performance, with rewards surpassing 3.0 by 3M and 5.0 by 3.4M timesteps. The agent consistently achieved average rewards above 7.0 after 4M timesteps, with a peak recorded at 8.87 at 4.5M timesteps. This performance translates to the agent reliably climbing 6 floors or more per episode.

The Q-values serve as a proxy for the agent's estimated future rewards. Initially, Q-values were low (~0.07), indicating uncertainty and lack of knowledge about the environment. After the agent's reward breakthrough, we observed a corresponding rise in Q-values. From 1.5M to 2.0M timesteps, Q-values increased from 0.11 to 0.23. By 4.5M, Q-values stabilized around 0.55 to 0.57, indicating the agent's increased confidence in its action-value estimates and suggesting convergence of learning.

The alignment between the rising average rewards and Q-values strongly suggests that the agent developed a meaningful policy for navigating the environment and achieving consistent progress. The breakthrough around 1.5M timesteps highlights the importance of exploration in environments with sparse rewards. Once the agent discovered reliable strategies for completing floors, its learning accelerated, reflected in the steep increase in both rewards and Q-values.

Our Rainbow DQN implementation demonstrates a substantial improvement over our initial PPO baseline, which failed to clear even a single floor consistently. Through a combination of prioritized replay, multi-step learning, and noisy networks, our Rainbow DQN agent learned to tackle the exploration problem of Obstacle Tower. The final agent achieved an average reward of 8.87, translating to 6 floors cleared on average, with peak runs exceeding this.

### Qualitative Analysis

Among the earlier models that we trained using the algorithms that we've described, it often struggles with making past the first floor. Even compared to the model that use a random policy, our agent is continuously jumping in every direction and walks around in loops within a section of the room. In these runs, the agent hasn't been able to reach the second floor because of the difficulty in navigating in the environment and making actions that prevents it from being stuck.

![example_agent_stuck](https://github.com/user-attachments/assets/b1058ff1-7f42-45dc-a706-5558fa6e8fc3)

There are occasions where the agent paths towards the door but doesn't fully go through and is walking back and forth. While we were experimenting with our methods, we were hoping the agent would consistently make its way to the doors, but it's repeated the same mistakes. After some period of time the agent even walks away from the door and ends up in the opposite side of the room. 

![example_agent_paths_to_door](https://github.com/user-attachments/assets/08b7196a-4a1d-41c0-b2a0-7c530c1a41f1)

In newer runs with PPO policy with rewards, and the Rainbow DQN models, the agent had learned to quickly go through floors. Using real-time replay, we were able to see the PPO with rewards model trek its way up through the floors and going through different rooms in the current floor to reach the next, showing that it was getting smarter. Although the PPO model had an average reward mean of almost 2 towards 2 million timesteps, we believe this model could go even further given more training time.

The Rainbow DQN model greatly excelled, as we trained in HPC3, we were able to extract a couple of demo videos in lower quality (as that is the low-quality environment we had ran it on the server because we were no longer doing it locally), and in one of this video, we saw the agent speed through floors 0-7, where it got stuck on floor 8 because the door was locked and it had not learned to go out of its path to pick up a key. Prior to this, keys to the doors that required them were layed upon in the path of the agent, so the agent never learned to explicitly go out of its way to collect those keys. Once again this agent was cut short, as it was scheduled to run 50 million timesteps, but only ran 5 million before it was shut down due to HCP3 issues. Given the promising results within this truncated training window, further experimentation with extended timesteps remains a critical future direction. Extended experimentation with longer training runs and enhanced exploration strategies may enable the agent to develop more sophisticated behaviors, such as backtracking for keys or handling more complex room layouts, potentially allowing it to climb even higher in the Obstacle Tower environment.

Visualization of Agent Behavior: Screenshots and videos show how the agent learns better strategies over time.

Failure Analysis: Identifies common mistakes, such as getting stuck in loops, misjudging jumps, or inefficient movements.

We plot performance graphs to illustrate the learning curve of different RL models.

## References
- [Obstacle Tower GitHub Repository](https://github.com/Unity-Technologies/obstacle-tower-env)
- [Obstacle Tower: A Generalization Challenge in Vision, Control, and Planning](https://arxiv.org/abs/1902.01378)
- [PPO Dash: Improving Generalization in Deep Reinforcement Learning](https://arxiv.org/abs/1907.06704)
- [Trying to navigate in the Obstacle Tower environment with Reinforcement Learning](https://smartcat.io/tech-blog/data-science/trying-to-navigate-in-the-obstacle-tower-environment-with-reinforcement-learning/)
- [ResNet Deep Learning: PyTorch Documentation](https://pytorch.org/vision/main/models/resnet.html)
- [Reinforcement Learning in Practice – Obstacle Tower Challenge](https://neurosys.com/blog/reinforcement-learning-obstacle-tower-challenge-2)
- [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://github.com/Kaixhin/Rainbow)

## AI Tool Usage
Throughout the development of our reinforcement learning agent in Obstacle Towers and learning different approaches we could use, we used the AI tool, ChatGPT, to simplify new algorithms that may improve upon the intial iterations of our trained models. We didn't have a profound foundation on what components of an algorithm could provide us with the outcomes we were looking for, so it helped tremendously being able to consult ChatGPT. We also relied on the tool to explain to us the affects of adjusting the hyperparameters for our PPO model which saved us a lot of time in figuring our through trial and error an optimal configuration. Additionally, when we consulted the professor about reward shaping, he mentioned that it's easy to have unexpected outcomes, so we made sure to cross check with ChatGPT whether the approaches we discussed were being clearly communicated to our agent.
