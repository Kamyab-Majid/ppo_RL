import gym
import envs
import numpy as np
from env.Helicopter import Helicopter
from env.controller import Controller

# my_data = np.genfromtxt('best_13_action_reward.csv', delimiter=',')


class state_finder:
    def __init__(self):
        self.my_heli = Helicopter()
        self.my_contr = Controller()
        self.ENV_ID = "CustomEnv-v0"
        # ENV_ID = "CartPole-v0"
        self.env = gym.make(self.ENV_ID)
        self.env.current_states = self.env.reset()

    def get_action(self, current_states, sl_action=[1, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, 1, 1]):
        action = self.my_contr.Controller_model(current_states, self.env.dt * self.env.counter, sl_action)
        return action


# obs = [0,0,0,0,0,0,0,0]
# [1, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, 1, 1]
# for i in range(25000):
#     # obs, reward, done, _ = env.step([my_data[i, 30],my_data[i, 31], my_data[i, 32], my_data[i, 33]])
#     action, reward = my_heli.Controller_model(env.current_states, [1, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, 1, 1],
#                                               env.dt * env.counter)
#     print(action)
#     obs, reward, done, _ = env.step(action.reshape(4))
#     if done:
#         print("done")
#         env.reset()
#         break

if __name__ == "__main__":
    abc = state_finder()
    random_states = np.random.uniform(-0.1, 0.1, 16)
    actions = abc.get_action(random_states)
