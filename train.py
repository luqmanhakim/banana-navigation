"""Contains a single train() function (for now) to train the given agent to the given unity environment.
"""

from collections import deque
import numpy as np
import torch


def train(agent, env, brain_name, score_to_stop=13.0, n_episodes=20000, max_t=1000,
          destn_path="./data/network-weights/network.pth",
          eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """ Train our given agent to the given environment.
    :param agent: a pytorch agent containing act() to react to the environment's state and step() to post-process it.
    :param env: a unity environment
    :param brain_name: name associated to the brain in `env`
    :param score_to_stop: average score to stop the training when reached.
    :param n_episodes: max number of episodes to trained - limited by `score_to_stop`.
    :param max_t: max steps in each episode
    Todo: move `destn_path` as an agents attribute
    :param destn_path: local path to store the agent's learned weights
    :param eps_start: starting epsilon
    :param eps_end: ending epsion
    :param eps_decay: decay rate for epsilon
    :return: list of scores from each episode
    """
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]  # get the current state
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state, reward = env_info.vector_observations[0], env_info.rewards[0]
            done = 1. if env_info.local_done[0] is True else 0.
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done == 1.:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= score_to_stop:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                         np.mean(scores_window)))
            torch.save(agent.local_network.state_dict(), destn_path)
            break
    return scores
