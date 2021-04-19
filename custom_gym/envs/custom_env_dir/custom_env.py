import sympy as sp
import numpy as np
from abc import ABC
import gym
from gym import spaces
from env.Helicopter import Helicopter
from env.controller import Controller


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
        self.symbolic_states_math, self.jacobian = self.My_helicopter.lambd_eq_maker(self.t, self.x_state, self.U_input)
        high = 100
        self.high_range = np.array([high for i in range(10)], dtype=np.float32)
        self.observation_space = spaces.Box(np.float32(-self.high_range), np.float32(self.high_range), dtype=np.float32)
        action_space_domain = {
            "deltacol": (-0.2, 1.2),
            "deltalat": (-0.2, 1.2),
            "deltalon": (-0.2, 1.2),
            "deltaped": (-0.2, 1.2),
            # "f1": (0.1, 5), "f2": (0.5, 20), "f3": (0.5, 20), "f4": (0.5, 10),
            # "lambda1": (0.5, 10), "lambda2": (0.1, 5), "lambda3": (0.1, 5), "lambda4": (0.1, 5),
            # "eta1": (0.2, 5), "eta2": (0.1, 5), "eta3": (0.1, 5), "eta4": (0.1, 5),
        }
        self.low_action_space = np.array(list(zip(*action_space_domain.values()))[0], dtype=np.float32)
        self.high_action_space = np.array(list(zip(*action_space_domain.values()))[1], dtype=np.float32)
        self.action_space = spaces.Box(low=self.low_action_space, high=self.high_action_space, dtype=np.float32)
        (
            self.t_start,
            self.dt,
            self.t_end,
        ) = (0, 0.01, 10)
        self.no_timesteps = int((self.t_end - self.t_start) / self.dt)
        self.all_t = np.linspace(self.t_start, self.t_end, self.no_timesteps)
        self.counter = 0
        self.best_reward = -10000000
        self.pn = 0
        self.counter_max = 0
        self.estim_per_step = 100
        self.longest_num_step = 0

    def step(self, current_action):
        # checking if the no_timesteps is met
        current_action = np.array(current_action)
        current_action = np.tanh(current_action)
        if self.counter >= self.no_timesteps-2:
            self.done = True
            print(self.done)
        # finding the new states
        control_input = self.current_action = tuple((current_action + 0.7) / 1.5)
        current_t = self.dt * self.counter
        try:
            self.current_states = self.My_helicopter.RK45(
                current_t,
                self.current_states,
                self.symbolic_states_math,
                self.dt,
                control_input,
            )
            self.current_states[0:3] = (1.42e-05, 7.31e-06, 4.29e-07)
            self.current_states[9:12] = (1.31e-09, 1.15e-07, 7.39e-07)
            self.all_obs[self.counter] = observation = list(self.current_states[3:9]) + list(self.current_states[12:16])
            error = -np.linalg.norm((abs(self.current_states[0:12]).reshape(12)), 1)
            self.all_rewards[self.counter] = self.control_rewards[self.counter] = error
        except OverflowError or ValueError or IndexError:
            self.jj = 1
        # finding the reward (error based on error-error_sm)
        reward = error

        # check to see if it is diverged
        if any(np.isnan(self.current_states)) or any(np.isinf(self.current_states)):
            self.jj = 3
        if self.jj > 0:
            print(self.jj)
            observation = self.all_obs[self.counter - 1]
            self.done = True
            self.counter -= 1
            self.all_rewards[self.counter] = reward = -10
        if self.done:
            current_num_step = np.count_nonzero(self.all_rewards)
            current_total_reward = sum(self.all_rewards)
            print(
                f"best reward: {self.best_reward}, longest step {self.longest_num_step} reward {current_total_reward}, step {current_num_step}"
            )
            if current_num_step < self.no_timesteps - 10:
                self.all_rewards[self.counter] = reward = reward - 7 * (1 - current_num_step / self.no_timesteps)
            if current_total_reward > self.best_reward and sum(self.all_rewards) != 0:
                if current_num_step >= self.longest_num_step:
                    self.longest_num_step = current_num_step
                    self.best_reward = sum(self.all_rewards)
                    np.savetxt(
                        "geekfile.txt",
                        np.c_[
                            self.all_t,
                            self.all_actions,
                            self.all_obs,
                            self.all_rewards,
                            self.control_rewards,
                        ],
                    )
        self.counter += 1
        return observation, float(reward), self.done, {}

    def reset(self):
        # initialization
        self.all_obs = np.zeros((self.no_timesteps, len(self.high_range)))
        self.all_actions = np.zeros((self.no_timesteps, len(self.high_action_space)))
        self.all_rewards = np.zeros((self.no_timesteps, 1))
        self.control_rewards = np.zeros((self.no_timesteps, 1))
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
        self.all_obs[self.counter] = observation = list(self.current_states[3:9]) + list(self.current_states[12:16])
        self.counter += 1
        self.done = False
        return observation
