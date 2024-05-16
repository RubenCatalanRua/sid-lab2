import gymnasium as gym
import numpy as np
import random
import time
import pandas as pd

from utils import draw_rewards, print_policy_for_taxi_env


class QLearningAgent:
    def __init__(
        self,
        env,
        gamma: float,
        learning_rate: float,
        epsilon: float,
        t_max: int,
        min_learning_rate: float,
        min_epsilon: float,
        learning_rate_decay: float,
        epsilon_decay: float,
    ):
        self.env = env
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.t_max = t_max
        self.min_learning_rate = min_learning_rate
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.learning_rate_decay = learning_rate_decay

    def select_action(self, state, training=True):
        if training and random.random() <= self.epsilon:
            return np.random.choice(self.env.action_space.n)
        else:
            return np.argmax(self.Q[state,])

    def update_Q(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state,])
        td_target = reward + self.gamma * self.Q[next_state, best_next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.learning_rate * td_error

    def learn_from_episode(self) -> int:
        state, _ = env.reset()
        total_reward = 0
        for i in range(self.t_max):
            action = self.select_action(state)
            new_state, new_reward, is_done, truncated, _ = self.env.step(action)
            total_reward += new_reward
            self.update_Q(state, action, new_reward, new_state)
            if is_done:
                break
            state = new_state
        self.decay_parameters()
        return total_reward

    def test_episode(self) -> int:
        state, _ = self.env.reset()
        t = 0
        total_reward = 0

        is_done = False
        while not is_done and t < self.t_max:
            action = self.select_action(state, False)
            state, reward, is_done, truncated, info = self.env.step(action)
            total_reward += reward
            t += 1
        return total_reward

    def policy(self):
        policy = np.zeros(env.observation_space.n)
        for s in range(env.observation_space.n):
            policy[s] = np.argmax(np.array(self.Q[s]))
        return policy

    def decay_parameters(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.learning_rate = max(
            self.min_learning_rate, self.learning_rate * self.learning_rate_decay
        )


if __name__ == "__main__":
    NUM_EPISODES = 1000
    T_MAX = 25  # Max number of steps in an episode

    # Q-learning parameters
    GAMMA = 0.95  # Discount factor (gamma): how much we value future rewards

    EPSILON = 1.0  # Exploration probability at the start of the training
    EPSILON_DECAY = 0.99
    MIN_EPSILON = 0.001

    LEARNING_RATE = 0.3
    LEARNING_RATE_DECAY = 0.99
    MIN_LEARNING_RATE = 0.01

    env = gym.make("Taxi-v3")

    average_time_per_episode_list = []
    total_training_time_list = []
    average_reward_obtained_test_list = []

    parameter_name = "epsilon"
    epsilons = [0.25, 0.5, 0.8, 1.0]

    for epsilon in epsilons:
        agent = QLearningAgent(
            env,
            gamma=GAMMA,
            learning_rate=LEARNING_RATE,
            epsilon=epsilon,
            t_max=T_MAX,
            min_learning_rate=MIN_LEARNING_RATE,
            min_epsilon=MIN_EPSILON,
            epsilon_decay=EPSILON_DECAY,
            learning_rate_decay=LEARNING_RATE_DECAY,
        )

        total_training_time = 0
        time_per_episode = []
        rewards = []
        for i in range(NUM_EPISODES):
            start_time = time.time()

            reward = agent.learn_from_episode()

            end_time = time.time()
            elapsed_time = end_time - start_time

            time_per_episode.append(elapsed_time)
            total_training_time += elapsed_time

            rewards.append(reward)
        # draw_rewards(rewards)

        print("hi")
        average_time_per_episode = np.mean(time_per_episode)

        NUM_TEST_EPISODES = 1000
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
            f"parameter_{parameter_name}": epsilons
        }
    )
    data.to_csv(f"./data_{parameter_name}.csv")
