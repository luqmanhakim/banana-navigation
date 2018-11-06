""" This program loads the agent's learned network trained in `train.py` and displays the play in the Banana env.
Note: Only works in iTerm (not sure why for now)
"""

from src.agent import DqnAgent
import torch
import time
from unityagents import UnityEnvironment

env = UnityEnvironment(file_name="./data/unity-banana-env/osx/Banana.app",
                       worker_id=1,
                       base_port=8095,  # change the base port as required
                       no_graphics=False)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

n_episodes = 10
max_t = 1000000
eps = 0.

agent = DqnAgent()
agent.local_network.load_state_dict(torch.load('./data/network-weights/default_at_13.pth', map_location='cpu'))

for i_episode in range(1, n_episodes+1):
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    for t in range(max_t):
        action = agent.act(state, eps)
        env_info = env.step(action)[brain_name]
        next_state, reward, done = env_info.vector_observations[0], \
                                   env_info.rewards[0], \
                                   1. if env_info.local_done[0] == True else 0.
        state = next_state
        score += reward
        print('\rEpisode {}\tScore: {:.2f}'.format(i_episode, score), end="")
        time.sleep(0.05)  # slow down the video abit
        if done == 1.:
            break

