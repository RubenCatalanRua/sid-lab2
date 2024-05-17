# To run QLearning experiments
python3 qlearning.py gamma 0.5 0.75 0.95 0.99
python3 qlearning.py epsilon 0.25 0.5 0.8 1.0
python3 qlearning.py learning_rate 0.01 0.1 0.25 0.5
python3 qlearning.py epsilon_decay 0.8 0.95 0.99 1.0
python3 qlearning.py learning_rate_decay 0.8 0.95 0.99 1.0


python3 valueIteration.py REWARD_THRESHOLD 2 3 4 5 6 7 7.5
python3 valueIteration.py GAMMA 0.8 0.85 0.9 0.95 0.99