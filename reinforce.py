import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import sys
from utils import save_rewards_plot, save_csv, save_metric_plot


class ReinforceAgent:
    def __init__(self, env, gamma, learning_rate, lr_decay=1, seed=0):
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        # Objeto que representa la política (J(theta)) como una matriz estados X acciones,
        # con una probabilidad inicial para cada par estado accion igual a: pi(a|s) = 1/|A|
        self.policy_table = np.ones(
            (self.env.observation_space.n, self.env.action_space.n)) / self.env.action_space.n
        np.random.seed(seed)

    def select_action(self, state, training=True):
        action_probabilities = self.policy_table[state]
        if training:
            # Escogemos la acción según el vector de policy_table correspondiente a la acción,
            # con una distribución de probabilidad igual a los valores actuales de este vector
            return np.random.choice(np.arange(self.env.action_space.n), p=action_probabilities)
        else:
            return np.argmax(action_probabilities)

    def update_policy(self, episode):
        states, actions, rewards = episode
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        loss = - \
            np.sum(
                np.log(self.policy_table[states, actions]) * discounted_rewards) / len(states)
        policy_logits = np.log(self.policy_table)
        for t in range(len(states)):
            G_t = discounted_rewards[t]
            action_probs = np.exp(policy_logits[states[t]])
            action_probs /= np.sum(action_probs)
            policy_gradient = G_t * (1 - action_probs[actions[t]])
            policy_logits[states[t], actions[t]
                          ] += self.learning_rate * policy_gradient
            # Alternativa:
            # policy_gradient = 1.0 / action_probs[actions[t]]
            # policy_logits[states[t], actions[t]] += self.learning_rate * G_t * policy_gradient
        exp_logits = np.exp(policy_logits)
        self.policy_table = exp_logits / \
            np.sum(exp_logits, axis=1, keepdims=True)
        return loss

    def learn_from_episode(self):
        state, _ = self.env.reset()
        T_MAX = 25
        episode = []
        done = False
        step = 0
        total_reward = 0
        while not done and step < T_MAX:
            action = self.select_action(state)
            next_state, reward, done, terminated, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            total_reward = total_reward + reward
            step = step + 1
        loss = self.update_policy(zip(*episode))
        self.learning_rate = self.learning_rate * self.lr_decay
        return total_reward, loss

    def policy(self):
        policy = np.zeros(env.observation_space.n)
        for s in range(env.observation_space.n):
            action_probabilities = self.policy_table[s]
            policy[s] = np.argmax(action_probabilities)
        return policy, self.policy_table


# CODIGO A CAMBIAR PARA CADA ALGORITMO
def create_agent(env, parameter_name: str, parameter_value) -> ReinforceAgent:
    T_MAX = 25  # Max number of steps in an episode

    # Reinforce parameters
    GAMMA = 0.95  # Discount factor (gamma): how much we value future rewards
    LEARNING_RATE = 0.3
    LEARNING_RATE_DECAY = 0.9
    MIN_LEARNING_RATE = 0.01

    parameters = {
        "gamma": GAMMA,
        "learning_rate": LEARNING_RATE,
        "lr_decay": LEARNING_RATE_DECAY,
    }

    if parameter_name in parameters:
        parameters[parameter_name] = parameter_value
    else:
        raise ValueError(f"Invalid parameter name: {parameter_name}")

    agent = ReinforceAgent(env, **parameters)
    return agent


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 reinforce.py <parameter_name> <parameter_value1> <parameter_value2> ... <parameter_valueN>")
        sys.exit(1)

    program_name = sys.argv[0].split('.')[0]
    parameter_name = sys.argv[1]
    parameter_values = [float(value) for value in sys.argv[2:]]

    print(f"----- Starting {program_name} with {parameter_name} = {parameter_values} -----")

    env = gym.make("Taxi-v3")

    NUM_EPISODES = 20000
    NUM_TEST_EPISODES = 1000

    average_time_per_episode_list = []
    total_training_time_list = []
    average_reward_obtained_test_list = []

    for parameter_value in parameter_values:
        print(f"Training with {parameter_name} = {parameter_value}")
        agent = create_agent(env, parameter_name, parameter_value)

        total_training_time = 0
        time_per_episode = []
        rewards = []
        for i in range(NUM_EPISODES):
            start_time = time.time()

            reward, _ = agent.learn_from_episode()

            end_time = time.time()
            elapsed_time = end_time - start_time

            time_per_episode.append(elapsed_time)
            total_training_time += elapsed_time

            rewards.append(reward)

        save_rewards_plot(program_name, parameter_name,
                          parameter_value, rewards)

        average_time_per_episode = np.mean(time_per_episode)
        T_MAX = 25
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
