import gym
import envs
import numpy as np
from env.Helicopter import Helicopter
from env.controller import Controller
from finding_random_states_and_actions import state_finder
import csv
# my_data = np.genfromtxt('best_13_action_reward.csv', delimiter=',')
my_state_finder = state_finder()
my_heli = Helicopter()
my_contr = Controller()
ENV_ID = "CustomEnv-v0"
my_env = gym.make(ENV_ID)
current_best_rew = -100000000000
import numpy
act_1 = numpy.linspace(0.1, 5.0, 10)
act_2 = numpy.linspace(0.1, 5.0, 10)
act_3 = numpy.linspace(5, 15, 10)
act_4 = numpy.linspace(5, 15, 10)
act_5 = numpy.linspace(2, 7, 10)
act_6 = numpy.linspace(2, 7, 10)
act_7 = numpy.linspace(0.1, 5, 10)
act_8 = numpy.linspace(0.1, 5, 10)
act_9 = numpy.linspace(0.1, 5, 10)
act_10 = numpy.linspace(0.1, 5, 10)
act_11 = numpy.linspace(0.1, 5, 10)
act_12 = numpy.linspace(0.1, 5.0, 10)
act_13 = numpy.linspace(0.1, 5.0, 10)
for i in range(len(act_1) - 1):
    act1 = act_1[i]
    for j in range(len(act_2) - 1):
        act2 = act_2[j]
        for k in range(len(act_3) - 1):
            act3 = act_3[k]
            for l in range(len(act_4) - 1):
                act4 = act_4[l]
                for m in range(len(act_5) - 1):
                    act5 = act_5[m]
                    for n in range(len(act_6) - 1):
                        act6 = act_6[n]
                        for o in range(len(act_7) - 1):
                            act7 = act_7[o]
                            for p in range(len(act_8) - 1):
                                act8 = act_8[p]
                                for q in range(len(act_9) - 1):
                                    act9 = act_9[q]
                                    for r in range(len(act_10) - 1):
                                        act10 = act_10[r]
                                        for s in range(len(act_11) - 1):
                                            act11 = act_11[s]
                                            for t in range(len(act_12) - 1):
                                                act12 = act_12[t]
                                                for u in range(len(act_13) - 1):
                                                    act13 = act_13[u]
                                                    sl_action = [act1,act2,act3,act4,act5,act6,act7,act8,act9,act10,act11,act12,act13]
                                                    print(sl_action)
                                                    done = False
                                                    my_env.reset()
                                                    while not done:
                                                        current_action = my_contr.Controller_model(my_env.current_states, my_env.dt * my_env.counter, action=sl_action)    
                                                        my_env.current_states, b, done, _ = my_env.step(current_action)
                                                    if my_env.best_reward > current_best_rew:
                                                        current_best_rew = my_env.best_reward
                                                        print("updated", sl_action)
                                                        with open("reward_step.csv", "a") as f:
                                                            writer = csv.writer(f)
                                                            writer.writerow(sl_action)
                                                    
                                                    
                                                        






