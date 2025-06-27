import os
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
    
    if args.use_baselines:
        print("Using Stable Baselines3 for training...")

        model = sb3.DQN(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            buffer_size=args.replay_buffer_capacity,
            batch_size=args.batch_size,
            target_update_interval=args.target_update_frequency,
            exploration_initial_eps=args.epsilon,
            exploration_final_eps=args.epsilon_min,
            verbose=1
        )

        model.learn(total_timesteps=args.max_steps)

        # Salvar modelo
        model_path = f"resultados/dqn_sb3_model_{args.run_name}.zip"
        model.save(model_path)
        print(f"Modelo SB3 salvo em: {model_path}")

        # Avaliação
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        # Registro de resultados
        os.makedirs("./resultados", exist_ok=True)
        with open("./resultados/test_results.txt", "a") as f:
            f.write(f"{args.run_name} [SB3]: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}\n\n")

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
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        replay_buffer_capacity=args.replay_buffer_capacity
    )

    agent = Agent(
        env=env,
        policy=dqn_policy,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        target_update_freq=args.target_update_frequency,
        run_name=args.run_name
    )

    if not args.test:
        mean_reward, std_reward = agent.train(max_steps=hyp["max_steps"], batch_size=hyp["batch_size"])
        env.close()
        print(f"Mean reward: {mean_reward} +/- {std_reward}")
        return

    if args.load_model_path:
        print(f"Carregando modelo de {args.load_model_path}")
        try:
            dqn_policy.load(args.load_model_path)
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            exit(1)
    mean_reward, std_reward = agent.test(episodes=100)
    env.close()
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
    with open(f"./resultados/test_results.txt", "a") as f:
        f.write(f"{args.run_name}: Mean reward: {mean_reward} +/- {std_reward}\n\n")


def load_hyperparameters(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    hyp = load_hyperparameters("hyperparameters.yml")

    parser = argparse.ArgumentParser()

    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--human", action="store_true", help="Run in human mode")
    parser.add_argument("--use-baselines", action="store_true", help="Use Stable Baselines3 for testing")
    parser.add_argument("--run_name", type=str, default="nome_padrao", help="Nome base para salvar resultados")
    parser.add_argument("--load-model-path", type=str, default=None,
                    help="Caminho para um modelo .pth existente a ser carregado antes do treinamento.")



    parser.add_argument("--learning-rate", type=float, default=hyp["learning_rate"], help="Learning rate for the DQN agent")
    parser.add_argument("--gamma", type=float, default=hyp["gamma"], help="Discount factor for the DQN agent")
    parser.add_argument("--epsilon", type=float, default=hyp["epsilon"], help="Initial epsilon for the DQN agent")
    parser.add_argument("--epsilon-decay", type=float, default=hyp["epsilon_decay"], help="Epsilon decay rate for the DQN agent")
    parser.add_argument("--epsilon-min", type=float, default=hyp["epsilon_min"], help="Minimum epsilon for the DQN agent")
    parser.add_argument("--target-update-frequency", type=int, default=hyp["target_update_frequency"], help="Target network update frequency")
    parser.add_argument("--replay-buffer-capacity", type=int, default=hyp["replay_buffer_capacity"], help="Replay buffer capacity for the DQN agent")
    parser.add_argument("--max-steps", type=int, default=hyp["max_steps"], help="Maximum number of steps for training")
    parser.add_argument("--batch-size", type=int, default=hyp["batch_size"], help="Batch size for training")

    args = parser.parse_args()

    main()