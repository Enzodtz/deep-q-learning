import torch as T
import numpy as np
from dqn import DeepQNetwork

from experience_replay import ExperienceReplay


class Agent:
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        input_dims,
        batch_size,
        n_actions,
        max_mem_size,
        eps_end,
        eps_dec,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]

        device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.Q_eval = DeepQNetwork(
            self.lr,
            n_actions=n_actions,
            input_dims=input_dims,
            device=device,
        )

        self.experience_replay = ExperienceReplay(
            input_dims, max_mem_size, batch_size, device
        )

    def store_transition(self, state, action, reward, state_, done):
        self.experience_replay.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if not self.experience_replay.has_min_entries():
            return

        self.Q_eval.optimizer.zero_grad()

        (
            state_batch,
            new_state_batch,
            reward_batch,
            terminal_batch,
            action_batch,
        ) = self.experience_replay.sample_batch()

        select_all_indexes = np.arange(
            self.experience_replay.batch_size, dtype=np.int32
        )

        q_eval = self.Q_eval.forward(state_batch)[select_all_indexes, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = (
            self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        )
