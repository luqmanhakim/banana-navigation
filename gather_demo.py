from train import train
from src.agent import DqnAgent
from unityagents import UnityEnvironment

""" Just a one-time script to gather weights at different episode checkpoints for demo purposes 
"""

env = UnityEnvironment(file_name="./data/unity-banana-env/linux/Banana.x86_64",
                       worker_id=3,
                       base_port=8089,
                       no_graphics=True)

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# default dqn agent
score_and_name = [(0.5, "05"),
                  (1.0, "1"),
                  (3.0, "3"),
                  (5.0, "5"),
                  (10.0, "10"),
                  (13.0, "13")]

for score, name in score_and_name:
    agent = DqnAgent()
    scores = train(agent, env, brain_name, n_episodes=10000, score_to_stop=score,
                   destn_path="./data/network-weights/default_at_{}.pth".format(name))
