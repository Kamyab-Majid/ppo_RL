import gym

from IPython.display import clear_output
import matplotlib.pyplot as plt


def make_env(env_name):
    def _thunk():
        env = gym.make(env_name)
        return env

    return _thunk


def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title("frame %s. reward: %s" % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()
