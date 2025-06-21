import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def plot_all_logs(results_dir="resultados", save_path="resultados/comparacao_resultados.png"):
    csv_files = glob.glob(os.path.join(results_dir, "train_log_*.csv"))
    
    if not csv_files:
        print("Nenhum arquivo train_log_*.csv encontrado.")
        return

    fig, axs = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    for file in csv_files:
        df = pd.read_csv(file)
        label = os.path.basename(file).replace("train_log_", "").replace(".csv", "")

        # Plot recompensa
        axs[0].plot(df["episode"], df["reward"], label=label)

        # Plot loss médio
        if "mean_loss" in df.columns:
            axs[1].plot(df["episode"], df["mean_loss"], label=label)

        # Plot epsilon
        if "epsilon" in df.columns:
            axs[2].plot(df["episode"], df["epsilon"], label=label)

    # Personalização dos gráficos
    axs[0].set_title("Recompensa por Episódio")
    axs[0].set_ylabel("Recompensa")

    axs[1].set_title("Loss Médio por Episódio")
    axs[1].set_ylabel("Loss")

    axs[2].set_title("Taxa de Exploração (ε) por Episódio")
    axs[2].set_ylabel("Epsilon")
    axs[2].set_xlabel("Episódio")

    for ax in axs:
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Gráfico salvo em: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_all_logs()
