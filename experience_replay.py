import numpy as np
import torch as T


class ExperienceReplay:
    def __init__(self, state_shape, max_mem_size, batch_size, device):
        self.mem_size = max_mem_size
        self.batch_size = batch_size

        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *state_shape), dtype=np.float32)
        self.new_state_memory = np.zeros(
            (self.mem_size, *state_shape), dtype=np.float32
        )
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.device = device

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_batch(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        state_batch = T.tensor(self.state_memory[batch]).to(self.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.device)
        action_batch = self.action_memory[batch]

        return state_batch, new_state_batch, reward_batch, terminal_batch, action_batch

    def has_min_entries(self):
        return self.mem_cntr >= self.batch_size
