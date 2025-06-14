import torch
import argparse
import yaml
import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3.common.evaluation import evaluate_policy
from agent import Agent
from dqn import DQN


def main():
    env = gym.make("LunarLander-v3", render_mode="human" if args.human else None)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open("hyperparameters.yml", "r") as file:
        hyp = yaml.safe_load(file)
    
    if args.use_baselines:
        print("Using Stable Baselines3 for testing...")
        model = sb3.DQN("MlpPolicy", env, verbose=1, learning_rate=hyp["learning_rate"], gamma=hyp["gamma"])
        model.learn(total_timesteps=hyp["max_steps"])
        model.save("dqn_sb3_model")
        
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        print(f"Mean reward: {mean_reward} +/- {std_reward}")

        if args.human:
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                env.render()

            env.close()
        return

    print("Using custom DQN implementation...")

    dqn_policy = DQN(
        state_size=state_size,
        action_size=action_size,
        device=device,
        learning_rate=hyp["learning_rate"],
        gamma=hyp["gamma"],
        replay_buffer_capacity=hyp["replay_buffer_capacity"]
    )

    agent = Agent(
        env=env, 
        policy=dqn_policy, 
        epsilon=hyp["epsilon"], 
        epsilon_decay=hyp["epsilon_decay"], 
        epsilon_min=hyp["epsilon_min"], 
        target_update_freq=hyp["target_update_frequency"]
    )

    if not args.test:
        mean_reward, std_reward = agent.train(max_steps=hyp["max_steps"], batch_size=hyp["batch_size"])
        env.close()
        print(f"Mean reward: {mean_reward} +/- {std_reward}")
        return

    dqn_policy.load("dqn_model.pth")
    mean_reward, std_reward = agent.test(episodes=100)
    env.close()
    print(f"Mean reward: {mean_reward} +/- {std_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--human", action="store_true", help="Run in human mode")
    parser.add_argument("--use-baselines", action="store_true", help="Use Stable Baselines3 for testing")
    args = parser.parse_args()

    main()