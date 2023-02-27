import torch as T
import numpy as np
from dqn import DeepQNetwork
import pickle

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
        online_to_target_frequency,
        save_frequency=500,
        filename="agent.pth",
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.online_to_target_frequency = online_to_target_frequency
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.filename = filename
        self.learning_counter = 0
        self.save_frequency = save_frequency

        device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.Q_online = DeepQNetwork(
            self.lr,
            n_actions=n_actions,
            input_dims=input_dims,
            device=device,
        )
        self.Q_target = DeepQNetwork(
            self.lr,
            n_actions=n_actions,
            input_dims=input_dims,
            device=device,
        )
        self.Q_target.load_from_network(self.Q_online)

        self.experience_replay = ExperienceReplay(
            input_dims, max_mem_size, batch_size, device
        )

    def store_transition(self, state, action, reward, state_, done):
        self.experience_replay.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = (
                T.tensor(observation, dtype=T.float32)
                .unsqueeze(0)  # adding batch dimension
                .to(self.Q_online.device)
            )
            actions = self.Q_online.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def save(self):
        with open(self.filename, "wb") as f:
            pickle.dump(self, f)
            print(f"[INFO] Saved model into {self.filename}")

    @classmethod
    def load(self, filename="agent.pth"):
        with open(filename, "rb") as f:
            obj = pickle.load(f)
            print(f"[INFO] Loaded model from {filename}")
            return obj

    def learn(self):
        if not self.experience_replay.has_min_entries():
            return

        self.Q_online.optimizer.zero_grad()

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

        # unsqueeze for adding channel
        q_eval = self.Q_online.forward(state_batch)[select_all_indexes, action_batch]
        q_next = self.Q_target.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_online.loss(q_target, q_eval).to(self.Q_online.device)
        loss.backward()
        self.Q_online.optimizer.step()

        self.epsilon = (
            self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        )

        self.learning_counter += 1
        if self.learning_counter % self.online_to_target_frequency == 0:
            self.Q_target.load_from_network(self.Q_online)
        if self.learning_counter % self.save_frequency == 0:
            self.save()
