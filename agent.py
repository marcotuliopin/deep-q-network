import numpy as np
from tqdm import tqdm


class Agent:
    def __init__(
        self,
        env,
        policy,
        epsilon=0.2,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        target_update_freq=10,
    ):
        self.env = env
        self.policy = policy
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq

    def train(self, max_steps: int, batch_size: int):
        steps = 0
        rewards = []

        while steps < max_steps:
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                if np.random.rand() <= self.epsilon: # Epsilon-greedy action selection
                    action = self.env.action_space.sample()
                else:
                    action = self.policy.act(state)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.policy.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
                rewards.append(reward)

                if len(self.policy.replay_buffer) > batch_size:
                    self.policy.replay(batch_size)

                # Update the target network every 10 episodes
                if steps % self.target_update_freq == 0:
                    self.policy.target_qnet.load_state_dict(self.policy.qnet.state_dict())

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            print(f"End of episode {steps}/{max_steps}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.4f}")

        self.policy.save("dqn_model.pth")
        return np.mean(rewards), np.std(rewards)

    def test(self, episodes):
        for episode in tqdm(range(episodes)):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.policy.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += reward

            print(f"Test Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")
