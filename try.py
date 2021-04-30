import csv
import gym
from env.Helicopter import Helicopter
from env.controller import Controller
from finding_random_states_and_actions import state_finder
my_state_finder = state_finder()
my_heli = Helicopter()
my_contr = Controller()
ENV_ID = "CustomEnv-v0"
my_env = gym.make(ENV_ID)
current_best_rew = -100000000000
# sl_action = [0.1, 0.1, 0.1, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
sl_action = [1, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, 1, 1]

print(sl_action)
done = False
my_env.current_states = my_env.reset()
while not done:
    current_action = my_contr.Controller_model(my_env.current_states, my_env.dt * my_env.counter, action=sl_action)   
    my_env.current_states, b, done, _ = my_env.step(current_action)
if my_env.best_reward > current_best_rew:
    current_best_rew = my_env.best_reward
    print("updated", sl_action)
    with open("reward_step.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(sl_action)
