# Usage

## Usage of programs


### Usage of Q-LEARNING program
To experiment with a parameter and different values for it, execute:
```
python3 qlearning.py <parameter_name> <parameter_value1> <parameter_value2> ... <parameter_valueN>
```
Where parameter_name can be:
- gamma
- epsilon
- epsilon_decay
- learning_rate
- learning_rate_decay

And parameter values are floats or integers.

NOTE: Only one parameter can be experimented with at a time, and all other
parameters will remain constant with a predefined value. In this case:
- GAMMA = 0.95
- EPSILON = 1.0
- EPSILON_DECAY = 0.99
- LEARNING_RATE = 0.3
- LEARNING_RATE_DECAY = 0.99

While these others are not supposed to be modified:
- NUM_EPISODES = 20000
- T_MAX = 20
- MIN_EPSILON = 0.001
- MIN_LEARNING_RATE = 0.01

### Direct Estimation
```
python3 direct_estimation.py
```
# sid-lab2
