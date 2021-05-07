import gym
import torch
from neural_network import ActorCritic
# import csv
from env.Helicopter import Helicopter
from env.controller import Controller
from finding_random_states_and_actions import state_finder

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
inputDim = 16  # takes variable 'x'
outputDim = 4  # takes variable 'y'
learningRate = 0.00001
epochs = 200000
hidden_size = 1000
model = ActorCritic(inputDim, outputDim, hidden_size).to(device)

model.load_state_dict(torch.load("model.pt"))
model.eval()


def trch_ft_device(input, device):
    output = torch.FloatTensor(input).to(device)
    return output


my_state_finder = state_finder()
my_heli = Helicopter()
my_contr = Controller()
ENV_ID = "CustomEnv-v0"
my_env = gym.make(ENV_ID)
current_best_rew = -100000000000
done = False
my_env.current_states = my_env.reset()
while not done:
    state = trch_ft_device(my_env.current_states, device)
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    dist, value = model(state)

    current_action = dist.sample()
    my_env.current_states, b, done, _ = my_env.step(list(current_action.numpy().squeeze()))
if my_env.best_reward > current_best_rew:
    current_best_rew = my_env.best_reward
    # print("updated", sl_action)
    # with open("reward_step.csv", "a") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(sl_action)
