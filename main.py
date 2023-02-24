import gym
from agent import Agent
import numpy as np

from utils import plot_learning_curve


ENV_NAME = "CartPole-v0"
ENV_PARAMS = {}
EPISODES = 500
LEARNING_RATE = 3e-3
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 9e-5
EPSILON_END = 1e-2
BATCH_SIZE = 64
MAX_MEMORY_SIZE = 100000
RENDER_TRAINING = False
RENDER_FPS = 60
# Edit network at dqn.py


if __name__ == "__main__":
    if RENDER_TRAINING:
        env = gym.make(ENV_NAME, **ENV_PARAMS)
        env.metadata["render_fps"] = RENDER_FPS
    else:
        env = gym.make(ENV_NAME, **ENV_PARAMS)

    agent = Agent(
        n_actions=env.action_space.n,
        input_dims=env.observation_space.shape,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        epsilon=EPSILON_START,
        eps_end=EPSILON_END,
        eps_dec=EPSILON_DECAY,
        batch_size=BATCH_SIZE,
        max_mem_size=MAX_MEMORY_SIZE,
    )
    scores, eps_history = [], []

    for i in range(EPISODES):
        score = 0
        done = False
        observation, _ = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)

            agent.learn()
            observation = observation_

        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print(
            f"Epsisode [{i}], Score [{score:.2f}], Avg Score [{avg_score:.2f}], Epsilon [{agent.epsilon:.3f}]",
        )

    plot_learning_curve(scores, eps_history)

    env = gym.make(ENV_NAME, render_mode="human", **ENV_PARAMS)
    while True:
        score = 0
        done = False
        observation, _ = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            score += reward

            observation = observation_

        print("Test score", score)
