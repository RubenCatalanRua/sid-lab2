# Usage

## Usage of programs
```
NOTE: Only one parameter can be experimented with at a time, and the parameter values are floats or integers. Graphics will be available in the folder 'data' after the program is executed.
	  
If you execute any of the programs incorrectly, an USAGE message will pop out, showing a correct input
```


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

Default values of the parameters are:
- NUM_EPISODES = 20000
- T_MAX = 20
- MIN_EPSILON = 0.001
- MIN_LEARNING_RATE = 0.01

### Usage of Direct Estimation program
To experiment with a parameter and different values for it, execute:
```
python3 direct_estimation.py <parameter_name> <parameter_value1> <parameter_value2> ... <parameter_valueN>
```
Where parameter_name can be:
- gamma
- num_trajectories
- reward_threshold

Default values of the parameters are:
- GAMMA = 0.95
- NUM_TRAJECTORIES = 1000
- REWARD_THRESHOLD = 0.9
- NUM_EPISODES = 100

### Usage of Value Iteration program
To experiment with a parameter and different values for it, execute:
```
python3 valueiteration.py <parameter_name> <parameter_value1> <parameter_value2> ... <parameter_valueN>
```
Where parameter_name can be:
- GAMMA
- REWARD_THRESHOLD

Default values of the parameters are:
- GAMMA = 0.9
- REWARD_THRESHOLD = 3
- NUM_EPISODES = 20000 (Non changeable)
- NUM_TEST_EPISODES = 1000 (Non changeable)

# sid-lab2
