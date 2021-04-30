# import csv
import gym
from env.Helicopter import Helicopter
from env.controller import Controller
from finding_random_states_and_actions import state_finder
from simple_pid import PID
import numpy as np
my_state_finder = state_finder()
my_heli = Helicopter()
my_contr = Controller()
ENV_ID = "CustomEnv-v0"
my_env = gym.make(ENV_ID)
current_best_rew = -100000000000

# print(sl_action)
done = False
# [0.8,0,0] lateral

k_bin = np.linspace(0, -10, num=10) 
i_bin = np.linspace(0, -10, num=3) 

current_action = [0, 0, 0, 0]
sl_action = [1, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, 1, 1]
pid_lat = PID(-2, 0, 0, setpoint=0)
# pid_lon = PID(0.2, 0, 0, setpoint=0)
# pid_col = PID(-2, -3, -0.3, setpoint=-1)
# pid_ped = PID(-1, -1, -0.5, setpoint=0)
for i in range((len(k_bin))):
    for j in range((len(i_bin))):
        kk = k_bin[i]
        ii = i_bin[j]
        # pid_lat = PID(kk, 0, ii, setpoint=0)
        pid_lon = PID(kk, ii, 0, setpoint=0)
        print(kk, ii)
        my_env.current_states = my_env.reset()
        done = False
        while not done:
            Yd, Ydotd, Ydotdotd, Y, Ydot = my_contr.Yposition(my_env.dt * my_env.counter, my_env.current_states)
            # ctrl = [float(deltacol), float(deltalat), float(deltalon), float(deltaped)]
            # pid_ped = PID(100, 1000, 0.05, setpoint=0)
            # print('Y             ', Y[2])
            sl_ctrl = my_contr.Controller_model(my_env.current_states, my_env.dt * my_env.counter, action=sl_action)
            # print('slaction      ', ['%.2e' % elem for elem in sl_ctrl])
            # sl_ctrl[0] = float(pid_col(Y[2]))
            # sl_ctrl[1] = float(pid_lat(Y[1]))
            sl_ctrl[2] = float(pid_lon(Y[0]))
            # sl_ctrl[3] = float(pid_ped(Y[3]))
            # print('upaction      ', ['%.2e' % elem for elem in sl_ctrl])
            my_env.current_states, b, done, _ = my_env.step(sl_ctrl)
            # print('current_states', my_env.current_states[2])
        if my_env.best_reward > current_best_rew:
            current_best_rew = my_env.best_reward
            # print("updated", sl_action)
            # with open("reward_step.csv", "a") as f:
            #     writer = csv.writer(f)
            #     writer.writerow(sl_action)

