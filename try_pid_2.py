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

k_bin = np.linspace(-1, 1, num=10)
i_bin = np.linspace(-1, 1, num=5)

current_action = [0, 0, 0, 0]
sl_action = [1, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, 1, 1]
# pid_lon = PID(kk, ii, 0, setpoint=-1.08e-01)
pid_lon = PID(0.2, 0.2, 0.01, setpoint=1.01e-04)  # controls q, theta
pid_lat = PID(0.2, 0.1, 0.01, setpoint=-1.08e-01)  # controls p, fi

pid_col = PID(-2, -3, -0.3, setpoint=0)
pid_ped = PID(-1, -1, -0.5, setpoint=-1.03e-03)
my_env.current_states = my_env.reset()
done = False
while not done:
    Yd, Ydotd, Ydotdotd, Y, Ydot = my_contr.Yposition(my_env.dt * my_env.counter, my_env.current_states)

    sl_ctrl = my_contr.Controller_model(my_env.current_states, my_env.dt * my_env.counter, action=sl_action)
    sl_ctrl[0] = float(pid_col(my_env.current_states[11]))
    sl_ctrl[1] = float(pid_lat(my_env.current_states[6]))
    sl_ctrl[2] = float(pid_lon(my_env.current_states[7]))
    sl_ctrl[3] = float(pid_ped(my_env.current_states[8]))
    states, b, done, _ = my_env.step(sl_ctrl)
    constant_dict = {
        "u": 1.,
        "v": 1.,
        "w": 0.,
        "p": 0.,
        "q": 0.,
        "r": 0.,
        "fi": 0.,
        "theta": 0.,
        "si": 0.,
        "x": 1.,
        "y": 1.,
        "z": 0.,
        "a": 0.,
        "b": 0.,
        "c": 0.,
        "d": 0.,
    }
    my_env.make_constant(list(constant_dict.values()))
if my_env.best_reward > current_best_rew:
    current_best_rew = my_env.best_reward
