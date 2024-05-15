import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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


def print_policy_for_taxi_env(policy):
    visual_help = {0: "v", 1: "^", 2: ">", 3: "<", 4: "P", 5: "D"}
    policy_arrows = [visual_help[x] for x in policy]
    print(np.array(policy_arrows).reshape([-1, 25]))
