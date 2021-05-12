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
        self.default_range = default_range = (-200, 200)
        self.t_start, self.dt, self.t_end = 0, 0.03, 2
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
            "t": (self.t_start, self.t_end),
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

        self.no_timesteps = int((self.t_end - self.t_start) / self.dt)
        self.all_t = np.linspace(self.t_start, self.t_end, self.no_timesteps)
        self.counter = 0
        self.best_reward = float("-inf")
        self.longest_num_step = 0
        self.reward_check_time = 0.7 * self.t_end
        self.high_action_diff = 0.2
        obs_header = str(list(self.observation_space_domain.keys()))[1:-1]
        act_header = str(list(self.action_space_domain.keys()))[1:-1]
        self.header = "time, " + act_header + ", " + obs_header + ", reward" + ", control reward"
        self.saver = save_files()
        self.reward_array = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), dtype=np.float32)
        self.reward_limit = [
            1.00e02,
            3.40e03,
            1.34e02,
            1.51e03,
            3.28e01,
            7.78e00,
            3.15e04,
            3.09e01,
            3.00e02,
            8.46e00,
            1.52e04,
            9.27e01,
        ]
        self.constant_dict = {
            "u": 0.0,
            "v": 0.0,
            "w": 0.0,
            "p": 1.0,
            "q": 1.0,
            "r": 0.0,
            "fi": 1.0,
            "theta": 1.0,
            "si": 0.0,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "a": 0.0,
            "b": 0.0,
            "c": 0.0,
            "d": 0.0,
        }
        self.save_counter = 0

    def action_wrapper(self, current_action) -> np.array:
        current_action = np.array(current_action)
        current_action = (
            current_action * (self.high_action_space - self.low_action_space) / 2
            + (self.high_action_space + self.low_action_space) / 2
        )
        self.all_actions[self.counter] = self.control_input = np.clip(
            current_action, self.low_action_space, self.high_action_space
        )

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
        self.all_obs[self.counter] = observation = list(np.append(self.current_states, self.counter * self.dt))
        return observation

    def reward_function(self) -> float:
        # add reward slope to the reward
        # TODO: normalizing reward
        # TODO: adding reward gap
        error = -np.linalg.norm((abs(self.current_states[0:12]).reshape(12)), 1)
        self.control_rewards[self.counter] = error
        for i in range(12):
            self.reward_array[i] = abs(self.current_states[i]) / self.default_range[1]
        reward = self.all_rewards[self.counter] = -sum(self.reward_array) + 0.17 / self.default_range[1]
        # control reward
        reward += 0.05 * float(
            self.control_rewards[self.counter] - self.control_rewards[self.counter - 1]
        )  # control slope
        reward += -0.005 * sum(abs(self.all_actions[self.counter]))  # input reward
        for i in (self.high_action_diff - self.all_actions[self.counter] - self.all_actions[self.counter - 1]) ** 2:
            reward += -min(0, i)
        return reward

    def check_diverge(self) -> bool:
        if max(abs(self.current_states)) > self.default_range[1]:
            print("state_diverge")
            self.jj = 1
            return True
        if self.counter >= self.no_timesteps - 1:  # number of timesteps
            return True
        # after self.reward_check_time it checks whether or not the reward is decreasing
        if self.counter > self.reward_check_time / self.dt:
            prev_time = int(self.counter - self.reward_check_time / self.dt)
            diverge_criteria = self.all_rewards[self.counter] - sum(self.all_rewards[0:prev_time]) / self.counter
            if diverge_criteria < -0.4:
                self.jj = 1
                return True
        bool_1 = any(np.isnan(self.current_states))
        bool_2 = any(np.isinf(self.current_states))
        if bool_1 or bool_2:
            self.jj = 1
            print("state_inf_nan_diverge")
        return False

    def done_jobs(self) -> None:
        counter = self.counter
        self.save_counter += 1
        current_total_reward = sum(self.all_rewards)
        if self.save_counter > 100:
            print("current_total_reward: ", current_total_reward)
            self.save_counter = 0
            self.saver.reward_step_save(self.best_reward, self.longest_num_step, current_total_reward, counter)
        if counter >= self.longest_num_step:
            self.longest_num_step = counter
        if current_total_reward >= self.best_reward and sum(self.all_rewards) != 0:
            self.best_reward = sum(self.all_rewards)
            ii = self.counter + 1
            self.saver.best_reward_save(
                self.all_t[0:ii],
                self.all_actions[0:ii],
                self.all_obs[0:ii],
                self.all_rewards[0:ii],
                self.control_rewards[0:ii],
                self.header,
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
            observation = np.append(observation, self.counter * self.dt)
            reward = self.min_reward
        if self.done:
            self.done_jobs()

        self.counter += 1
        self.make_constant(list(self.constant_dict.values()))
        return observation, reward, self.done, {}

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
        self.initial_states = [
            3.70e-04,  # 0u
            1.15e-02,  # 1v
            4.36e-04,  # 2w
            -5.08e-03,  # 3p
            2.04e-04,  # 4q
            2.66e-05,  # 5r
            -1.08e-01,  # 6fi
            1.01e-04,  # 7theta
            -1.03e-03,  # 8si
            -4.01e-05,  # 9x
            -5.26e-02,  # 10y
            -2.94e-04,  # 11z
            -4.36e-06,  # 12a
            -9.77e-07,  # 13b
            -5.66e-05,  # 14c
            7.81e-04,
        ]  # 15d
        self.current_states = self.initial_states = list((np.random.rand(16) * 0.02 - 0.01))
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.all_obs[self.counter] = observation = np.append(self.current_states, self.counter * self.dt)
        self.done = False
        return observation

    def make_constant(self, true_list):
        for i in range(len(true_list)):
            if i == 1:
                self.current_states[i] = self.initial_states[i]
