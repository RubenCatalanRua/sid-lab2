import gymnasium as gym
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import collections
import time
import sys
from utils import draw_rewards,save_rewards_plot, save_csv

class DirectEstimationAgent:
    def __init__(self, env, gamma, num_trajectories, reward_threshold, t_max:int):
        self.env = env
        self.state, _ = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.V = np.zeros(self.env.observation_space.n)
        self.gamma = gamma
        self.num_trajectories = num_trajectories
        self.reward_threshold = reward_threshold
        self.t_max = t_max

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, truncated, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            if is_done:
                self.state, _ = self.env.reset()
            else:
                self.state = new_state

    def calc_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for s_, count in target_counts.items():
            r = self.rewards[(state, action, s_)]
            prob = (count / total)
            action_value += prob*(r + self.gamma * self.V[s_])
        return action_value

    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def value_iteration(self):
        self.play_n_random_steps(self.num_trajectories)
        max_diff = 0
        for state in range(self.env.observation_space.n):
            state_values = [
                self.calc_action_value(state, action)
                for action in range(self.env.action_space.n)
            ]
            new_V = max(state_values)
            diff = abs(new_V - self.V[state])
            if diff > max_diff:
                max_diff = diff
            self.V[state] = new_V
        return self.V, max_diff

    def policy(self):
        policy = np.zeros(env.observation_space.n)
        for s in range(env.observation_space.n):
            Q_values = [self.calc_action_value(
                s, a) for a in range(self.env.action_space.n)]
            policy[s] = np.argmax(np.array(Q_values))
        return policy
    
    def test_episode(self) -> int:
        state, _ = self.env.reset()
        t = 0
        total_reward = 0

        is_done = False
        for i in range(int(agent.t_max)):
            action = self.select_action(state)
            state, reward, is_done, truncated, info = self.env.step(action)
            total_reward += reward
            t += 1
        return total_reward

def check_improvements():
    reward_test = 0.0
    for i in range(NUM_EPISODES):
        total_reward = 0.0
        state, _ = env.reset()
        for i in range(int(agent.t_max)):
            action = agent.select_action(state)
            new_state, new_reward, is_done, truncated, _ = env.step(action)
            total_reward += new_reward
            if is_done:
                break
            state = new_state
        reward_test += total_reward
    reward_avg = reward_test / NUM_EPISODES
    return reward_avg

    
def train(agent):
    rewards = []
    max_diffs = []
    t = 0
    best_reward = 0.0

    while best_reward < agent.reward_threshold:
        _, max_diff = agent.value_iteration()
        max_diffs.append(max_diff)
        # print("After value iteration, max_diff = " +str(max_diff))
        t += 1
        reward_test = check_improvements()
        rewards.append(reward_test)

        if reward_test > best_reward:
            print(f"Best reward updated {reward_test:.2f} at iteration {t}")
            best_reward = reward_test

    return rewards, max_diffs

# def draw_rewards(rewards, type):
#     data = pd.DataFrame(
#         {'Episode': range(1, len(rewards) + 1), 'Reward': rewards})
#     plt.figure(figsize=(10, 6))
#     sns.lineplot(x='Episode', y='Reward', data=data)

#     plt.title('Rewards Over Episodes')
#     plt.xlabel('Episode')
#     plt.ylabel('Reward')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig('rewards_plot.png')


# CODIGO A CAMBIAR PARA CADA ALGORITMO
def create_agent(env, parameter_name: str, parameter_value) -> DirectEstimationAgent:
    T_MAX = 25  # Max number of steps in an episode

    # Model based parameters
    GAMMA = 0.95  # Discount factor (gamma): how much we value future rewards
    NUM_TRAJECTORIES = 10
    REWARD_THRESHOLD = 0.9 
    
    parameters = {
        "gamma": GAMMA,
        "num_trajectories": NUM_TRAJECTORIES,
        "t_max": T_MAX,
        "reward_threshold": REWARD_THRESHOLD,
    }

    if parameter_name in parameters:
        parameters[parameter_name] = parameter_value
    else:
        raise ValueError(f"Invalid parameter name: {parameter_name}")

    agent = DirectEstimationAgent(env, **parameters)
    return agent

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 modelBased.py <parameter_name> <parameter_value1> <parameter_value2> ... <parameter_valueN>")
        sys.exit(1)

    program_name = sys.argv[0].split('.')[0]
    parameter_name = sys.argv[1]
    parameter_values = [float(value) for value in sys.argv[2:]]

    env = gym.make("Taxi-v3")

    NUM_EPISODES = 1
    NUM_TEST_EPISODES = 1

    average_time_per_episode_list = []
    total_training_time_list = []
    average_reward_obtained_test_list = []

    for parameter_value in parameter_values:
        agent = create_agent(env, parameter_name, parameter_value,)

        total_training_time = 0
        time_per_episode = []
        rewards = []
        for i in range(NUM_EPISODES):
            start_time = time.time()

            reward,_ = train(agent)
            
            end_time = time.time()
            elapsed_time = end_time - start_time

            time_per_episode.append(elapsed_time)
            total_training_time += elapsed_time

            rewards.append(np.mean(reward))
        # print(rewards)
        # draw_rewards(rewards)
        save_rewards_plot(program_name, parameter_name, parameter_value, rewards)

        average_time_per_episode = np.mean(time_per_episode)

        test_rewards = []
        for i in range(NUM_TEST_EPISODES):
            reward = agent.test_episode()
            test_rewards.append(reward)
        average_reward_obtained_test = np.mean(test_rewards)

        average_time_per_episode_list.append(average_time_per_episode)
        total_training_time_list.append(total_training_time)
        average_reward_obtained_test_list.append(average_reward_obtained_test)

    data = pd.DataFrame(
        {
            "average_time_per_episode": average_time_per_episode_list,
            "total_training_time": total_training_time_list,
            "average_reward_obtained_test": average_reward_obtained_test_list,
            f"{parameter_name}": parameter_values,
        }
    )
    save_csv(program_name, parameter_name, data)






    # # env = gym.make("Taxi-v3", render_mode="human")
    # env = gym.make("Taxi-v3")
    # agent = DirectEstimationAgent(env, gamma=GAMMA, num_trajectories=25)
    # train(agent)

    # is_done = False
    # rewards = []
    # for n_ep in range(NUM_EPISODES):
    #     state, _ = env.reset()
    #     print('Episode: ', n_ep)
    #     total_reward = 0
    #     for i in range(T_MAX):
    #         action = agent.select_action(state)
    #         state, reward, is_done, truncated, _ = env.step(action)
    #         total_reward = total_reward + reward
    #         env.render()
    #         if is_done:
    #             break
    #     rewards.append(total_reward)
    # draw_rewards(rewards, 'rewards')
    # mean_rewards = np.mean(rewards, axis=0)
    # print(f"Mean reward: {mean_rewards:.2f}")

    # NUM_TEST_EPISODES = 1000
    # test_rewards = []
    # for i in range(NUM_TEST_EPISODES):
    #     print(f"Test episode {i}")
    #     reward = agent.test_episode()
    #     test_rewards.append(reward)
    # print (f"Average reward over {NUM_TEST_EPISODES} test episodes: {np.mean(test_rewards)}")
    # # draw_rewards(mean_rewards, 'mean')
