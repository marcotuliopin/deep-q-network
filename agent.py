import os
import csv
import time
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
        run_name="default_name",
    ):
        self.env = env
        self.policy = policy
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq

        self.run_name = f"{run_name}".replace('.', '')


    def train(self, max_steps: int, batch_size: int):
        steps = 0
        rewards = []
        episode = 0
        start_time = time.time()
        log_time = time.time()
        n_updates = 0
        losses = []

        # Criar pasta e arquivo CSV para salvar os resultados
        os.makedirs("resultados", exist_ok=True)
        log_path = os.path.join("resultados", f"train_log_{self.run_name}.csv")
        with open(log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "length", "epsilon", "mean_loss", "steps"])

        while steps < max_steps:
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            episode_length = 0

            while not done:
                if np.random.rand() <= self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = self.policy.act(state)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Save the experience in the replay buffer
                self.policy.remember(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                steps += 1
                episode_length += 1

                # Learn from the replay buffer
                if len(self.policy.replay_buffer) > batch_size:
                    loss = self.policy.replay(batch_size)
                    losses.append(loss)
                    n_updates += 1

                # Update target network
                if steps % self.target_update_freq == 0:
                    self.policy.target_qnet.load_state_dict(self.policy.qnet.state_dict())

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            episode += 1
            mean_loss = 0.0 if not losses else np.mean(losses[-100:])
            fps = steps / (time.time() - start_time)

            # Salvar log no CSV
            with open(log_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([episode, total_reward, episode_length, self.epsilon, mean_loss, steps])

            # Print log every 2 seconds or at the end of training
            if time.time() - log_time > 2 or steps >= max_steps:
                self._print_log(
                    episode=episode,
                    total_steps=steps,
                    episode_reward=total_reward,
                    episode_length=episode_length,
                    epsilon=self.epsilon,
                    loss=mean_loss,
                    learning_rate=self.policy.optimizer.param_groups[0]['lr'],
                    n_updates=n_updates,
                    fps=int(fps),
                    time_elapsed=time.time() - start_time
                )
                log_time = time.time()
        rewards.append(total_reward)
        #self.policy.save("dqn_model.pth")
        model_path = os.path.join("resultados", f"dqn_model_{self.run_name}.pth")
        self.policy.save(model_path)

        return np.mean(rewards), np.std(rewards)


    def test(self, episodes):
        rewards = []
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
            rewards.append(total_reward)
            print(f"Test Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")
        return np.mean(rewards), np.std(rewards)
    
    @staticmethod
    def _print_log(
        episode,
        total_steps,
        episode_reward,
        episode_length,
        epsilon,
        loss,
        learning_rate,
        n_updates,
        fps,
        time_elapsed
    ):
        print("----------------------------------")
        print(f"| rollout/            |          |")
        print(f"|    ep_len_mean      | {episode_length:<8} |")
        print(f"|    ep_rew_mean      | {episode_reward:<8.2f} |")
        print(f"|    exploration_rate | {epsilon:<8.3f} |")
        print(f"| time/               |          |")
        print(f"|    episodes         | {episode:<8} |")
        print(f"|    fps              | {fps:<8} |")
        print(f"|    time_elapsed     | {time_elapsed:<8.1f} |")
        print(f"|    total_timesteps  | {total_steps:<8} |")
        print(f"| train/              |          |")
        print(f"|    learning_rate    | {learning_rate:<8.6f} |")
        print(f"|    loss             | {loss:<8.4f} |")
        print(f"|    n_updates        | {n_updates:<8} |")
        print("----------------------------------")
