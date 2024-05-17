import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import sys
from utils import save_rewards_plot, save_csv, save_metric_plot


def test_episode(agent, env):
    env.reset()
    is_done = False
    t = 0

    while not is_done:
        action = agent.select_action()
        state, reward, is_done, truncated, info = env.step(action)
        t += 1
    return state, reward, is_done, truncated, info


class ValueIterationAgent:
    def __init__(self, env, gamma):
        self.env = env
        self.V = np.zeros(self.env.observation_space.n)
        self.gamma = gamma

    def calc_action_value(self, state, action):
        action_value = sum([prob * (reward + self.gamma * self.V[next_state])
                            for prob, next_state, reward, _
                            in self.env.unwrapped.P[state][action]])
        return action_value

    def select_action(self, state):
        best_action = best_value = None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if not best_value or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def value_iteration(self):
        max_diff = 0
        for state in range(self.env.observation_space.n):
            state_values = []
            for action in range(self.env.action_space.n):
                state_values.append(self.calc_action_value(state, action))
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


def check_improvements():
    reward_test = 0.0
    time_per_episode = []
    total_training_time = 0
    for i in range(NUM_EPISODES):
        start_time = time.time()
        total_reward = 0.0
        state, _ = env.reset()
        for i in range(T_MAX):
            action = agent.select_action(state)
            new_state, new_reward, is_done, truncated, _ = env.step(action)
            total_reward += new_reward
            if is_done:
                break
            state = new_state
        reward_test += total_reward
        end_time = time.time()
        elapsed_time = end_time - start_time
        time_per_episode.append(elapsed_time)
        total_training_time += elapsed_time

    reward_avg = reward_test / NUM_EPISODES
    return reward_avg, time_per_episode, total_training_time


def train(agent):
    rewards = []
    max_diffs = []
    t = 0
    best_reward = 0.0
    total_training_time = 0
    time_per_episode = []

    while best_reward < REWARD_THRESHOLD:
        _, max_diff = agent.value_iteration()
        max_diffs.append(max_diff)
        print("After value iteration, max_diff = " + str(max_diff))
        t += 1
        reward_test,time_test,training_time_test = check_improvements()
        rewards.append(reward_test)
        total_training_time += training_time_test
        time_per_episode += time_test
        if reward_test > best_reward:
            best_reward = reward_test

    return rewards, max_diffs, time_per_episode, total_training_time


# Fixed variables
T_MAX = 25                  # Max steps over an episode
NUM_EPISODES = 20000        # The total number of episodes
# Number of test episodes (we calculate the median of its results)
NUM_TEST_EPISODES = 2000


# Tested variables (these are default values, arguments will override them)
GAMMA = 0.9                 # How much we value future rewards
# Instead of testing for convergence, we use a reward threshold which we must overcome
REWARD_THRESHOLD = 3

# No render_mode, for faster execution
env = gym.make("Taxi-v3")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 valueIteration.py <parameter_name> <parameter_value1> <parameter_value2> ... <parameter_valueN>")
        print("where <parameter_name> == (GAMMA{0..1} | REWARD_THRESHOLD)")
        sys.exit(1)
    if sys.argv[1] not in ["GAMMA", "REWARD_THRESHOLD"]:
        print("<parameter_name> == (GAMMA{0..1} | REWARD_THRESHOLD)")
        sys.exit(1)

    program_name = sys.argv[0].split('.')[0]
    parameter_name = sys.argv[1]
    parameter_values = [float(value) for value in sys.argv[2:]]   

    average_time_per_episode_list = []
    total_training_time_list = []
    average_reward_obtained_test_list = []


    print(f"----- Starting {program_name} with {parameter_name} = {parameter_values} -----")


    for parameter_value in parameter_values:
        print(f"Training with {parameter_name} = {parameter_value}")

        if parameter_name == "GAMMA": GAMMA = parameter_value
        else: REWARD_THRESHOLD = parameter_value

        is_done = False
        rewards = []

        # Creamos el agente
        agent = ValueIterationAgent(env, gamma=GAMMA)

        rewards, _, time_per_episode, total_training_time = train(agent)

        save_rewards_plot(program_name, parameter_name, parameter_value, rewards)

        average_time_per_episode = np.mean(time_per_episode)

        test_rewards = []
        for n_test in range(NUM_TEST_EPISODES):
                state, _ = env.reset()
                print('Episode_test: ', n_test)
                total_reward = 0
                for i in range(T_MAX):
                    action = agent.select_action(state)
                    state, reward, is_done, truncated, _ = env.step(action)
                    total_reward = total_reward + reward
                    env.render()
                    if is_done:
                        break
                test_rewards.append(total_reward)

        average_reward_obtained_test = np.mean(test_rewards)
        average_time_per_episode_list.append(average_time_per_episode)
        total_training_time_list.append(total_training_time)
        average_reward_obtained_test_list.append(average_reward_obtained_test)
        print(f"Finished training")
    data = pd.DataFrame(
        {
            "average_time_per_episode": average_time_per_episode_list,
            "total_training_time": total_training_time_list,
            "average_reward_obtained_test": average_reward_obtained_test_list,
            f"{parameter_name}": parameter_values,
        }
    )
    save_csv(program_name, parameter_name, data)
    save_metric_plot(program_name, parameter_name, data)
