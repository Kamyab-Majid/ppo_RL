import sympy as sp
import numpy as np
from abc import ABC
import gym
from gym import spaces
from env.Helicopter import Helicopter
from env.controller import Controller
from utils import save_files


class CustomEnv(gym.Env, ABC):
    def __init__(self):
        self.U_input = [U1, U2, U3, U4] = sp.symbols("U1:5", real=True)
        self.x_state = [
            u_velocity,
            v_velocity,
            w_velocity,
            p_angle,
            q_angle,
            r_angle,
            fi_angle,
            theta_angle,
            si_angle,
            xI,
            yI,
            zI,
            a_flapping,
            b_flapping,
            c_flapping,
            d_flapping,
        ] = sp.symbols("x1:17", real=True)
        self.My_helicopter = Helicopter()
        self.My_controller = Controller()
        self.t = sp.symbols("t")
        self.symbolic_states_math, jacobian = self.My_helicopter.lambd_eq_maker(self.t, self.x_state, self.U_input)
        self.default_range = default_range = (-20, 20)
        self.observation_space_domain = {
            "u_velocity": default_range,
            "v_velocity": default_range,
            "w_velocity": default_range,
            "p_angle": default_range,
            "q_angle": default_range,
            "r_angle": default_range,
            "fi_angle": default_range,
            "theta_angle": default_range,
            "si_angle": default_range,
            "xI": default_range,
            "yI": default_range,
            "zI": default_range,
            "a_flapping": default_range,
            "b_flapping": default_range,
            "c_flapping": default_range,
            "d_flapping": default_range,
        }
        self.low_obs_space = np.array(list(zip(*self.observation_space_domain.values()))[0], dtype=np.float32)
        self.high_obs_space = np.array(list(zip(*self.observation_space_domain.values()))[1], dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_obs_space, high=self.high_obs_space, dtype=np.float32)
        self.default_act_range = default_act_range = (-0.3, 0.3)

        self.action_space_domain = {
            "deltacol": default_act_range,
            "deltalat": default_act_range,
            "deltalon": default_act_range,
            "deltaped": default_act_range,
            # "f1": (0.1, 5), "f2": (0.5, 20), "f3": (0.5, 20), "f4": (0.5, 10),
            # "lambda1": (0.5, 10), "lambda2": (0.1, 5), "lambda3": (0.1, 5), "lambda4": (0.1, 5),
            # "eta1": (0.2, 5), "eta2": (0.1, 5), "eta3": (0.1, 5), "eta4": (0.1, 5),
        }
        self.low_action_space = np.array(list(zip(*self.action_space_domain.values()))[0], dtype=np.float32)
        self.high_action_space = np.array(list(zip(*self.action_space_domain.values()))[1], dtype=np.float32)
        self.action_space = spaces.Box(low=self.low_action_space, high=self.high_action_space, dtype=np.float32)
        self.min_reward = -13
        self.t_start, self.dt, self.t_end = 0, 0.005, 2
        self.no_timesteps = int((self.t_end - self.t_start) / self.dt)
        self.all_t = np.linspace(self.t_start, self.t_end, self.no_timesteps)
        self.counter = 0
        self.best_reward = -100_000_000
        self.longest_num_step = 0
        self.reward_check_time = 0.7
        obs_header = str(list(self.observation_space_domain.keys()))[1:-1]
        act_header = str(list(self.action_space_domain.keys()))[1:-1]
        self.header = "time, " + act_header + ", " + obs_header + ", reward" + ", control reward"
        self.saver = save_files()
        self.reward_array = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), dtype=np.float32)

    def action_wrapper(self, current_action) -> np.array:
        current_action = np.array(current_action)
        self.all_actions[self.counter] = self.control_input = \
            np.clip(current_action, self.default_act_range[0], self.default_act_range[1])

    def find_next_state(self) -> list:
        current_t = self.dt * self.counter
        self.current_states = self.My_helicopter.RK45(
            current_t,
            self.current_states,
            self.symbolic_states_math,
            self.dt,
            self.control_input,
        )

    def observation_function(self) -> list:
        self.all_obs[self.counter] = observation = list(self.current_states)
        return observation

    def reward_function(self):
        # add reward slope to the reward
        error = -np.linalg.norm((abs(self.current_states[0:12]).reshape(12)), 1)
        self.control_rewards[self.counter] = error
        for i in range(12):
            self.reward_array[i] = abs(self.current_states[i]) / self.default_range[1]
        reward = self.all_rewards[self.counter] = -sum(self.reward_array) + 0.17 / self.default_range[1]
        reward += 0.4 * (reward - self.all_rewards[self.counter - 1])  
        return reward

    def check_diverge(self) -> bool:
        if max(abs(self.current_states)) > self.default_range[1]:
            print('state_diverge')
            self.jj = 1
            return True        
        if self.counter >= self.no_timesteps - 1:  # number of timesteps
            print('successful')
            return True
        # after self.reward_check_time it checks whether or not the reward is decreasing
        if self.counter > self.reward_check_time / self.dt:
            if self.all_rewards[self.counter] - \
                    self.all_rewards[int(self.counter - self.reward_check_time / self.dt)] < 0:
                print('reward_diverge')
                self.jj = 1
                return True
        bool_1 = any(np.isnan(self.current_states))
        bool_2 = any(np.isinf(self.current_states))
        if bool_1 or bool_2:
            self.jj = 1
            print('state_inf_nan_diverge')
        return False           

    def done_jobs(self) -> None:
        current_num_step = self.counter
        current_total_reward = sum(self.all_rewards)
        if current_num_step > 10:
            self.saver.reward_step_save(
                self.best_reward, self.longest_num_step, current_total_reward, current_num_step
            )
        if current_num_step >= self.longest_num_step:
            self.longest_num_step = current_num_step
        if current_total_reward > self.best_reward and sum(self.all_rewards) != 0:
            self.best_reward = sum(self.all_rewards)
            self.saver.best_reward_save(
                self.all_t, self.all_actions, self.all_obs, self.all_rewards, self.control_rewards, self.header
            )

    def step(self, current_action):
        self.action_wrapper(current_action)
        try:
            self.find_next_state()
        except OverflowError or ValueError or IndexError:
            self.jj = 1
        observation = self.observation_function()
        reward = self.reward_function()
        self.done = self.check_diverge()
        if self.jj == 1:
            observation = list((self.all_obs[self.counter]) - self.default_range[0])
            reward = self.min_reward
        if self.done:
            self.done_jobs()

        self.counter += 1
        return observation, float(reward), self.done, {}

    def reset(self):
        # initialization
        self.all_obs = np.zeros((self.no_timesteps, len(self.high_obs_space)))
        self.all_actions = np.zeros((self.no_timesteps, len(self.high_action_space)))
        self.all_rewards = np.zeros((self.no_timesteps, 1))
        self.control_rewards = np.zeros((self.no_timesteps, 1))
        self.control_input = np.array((0, 0, 0, 0), dtype=np.float32)
        self.jj = 0
        self.counter = 0
        # Yd, Ydotd, Ydotdotd, Y, Ydot = self.My_controller.Yposition(0, self.current_states)
        self.current_states = [
            1.42e-05,
            7.31e-06,
            4.29e-07,
            -4.98e-06,
            -5.02e-07,
            1.70e-06,
            -1.09e-01,
            6.55e-07,
            -7.89e-03,
            -2.15e-03,
            -9.05e-02,
            -9.40e-02,
            1.58e-09,
            1.31e-09,
            1.15e-07,
            7.39e-07,
        ]
        self.all_obs[self.counter] = observation = self.current_states
        self.done = False
        return observation
