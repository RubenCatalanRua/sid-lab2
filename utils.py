import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def draw_rewards(rewards: list[int]):
    data = pd.DataFrame({"Episode": range(1, len(rewards) + 1), "Reward": rewards})
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="Episode", y="Reward", data=data)

    plt.title(f"Rewards Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()

    # plt.savefig(f"rewards_plot_{parameters}.png")

    plt.show()


def save_rewards_plot(
    program_name: str, parameter_name: str, parameter_value, rewards: list[int]
):
    data = pd.DataFrame({"Episode": range(1, len(rewards) + 1), "Reward": rewards})
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="Episode", y="Reward", data=data)

    plt.title(f"Rewards Over Episodes for {parameter_name} = {parameter_value}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()

    if not os.path.exists(f"data"):
        os.makedirs(f"data")

    if not os.path.exists(f"data/{program_name}"):
        os.makedirs(f"data/{program_name}")

    if not os.path.exists(f"data/{program_name}/{parameter_name}"):
        os.makedirs(f"data/{program_name}/{parameter_name}")

    plt.savefig(
        f"data/{program_name}/{parameter_name}/rewards_plot_{parameter_name}_{parameter_value}.png"
    )


def save_csv(program_name: str, parameter_name: str, data: pd.DataFrame):
    if not os.path.exists(f"data"):
        os.makedirs(f"data")

    if not os.path.exists(f"data/{program_name}"):
        os.makedirs(f"data/{program_name}")

    if not os.path.exists(f"data/{program_name}/{parameter_name}"):
        os.makedirs(f"data/{program_name}/{parameter_name}")

    data.to_csv(f"data/{program_name}/{parameter_name}/data.csv")


def print_policy_for_taxi_env(policy):
    visual_help = {0: "v", 1: "^", 2: ">", 3: "<", 4: "P", 5: "D"}
    policy_arrows = [visual_help[x] for x in policy]
    print(np.array(policy_arrows).reshape([-1, 25]))
