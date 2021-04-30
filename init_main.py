import numpy as np
import torch
from torch.autograd import Variable
# import matplotlib.pyplot as plt
from init_actor_critic import ActorCritic
from finding_random_states_and_actions import state_finder

# create dummy data for training
# xx_1 = (np.linspace(2,27,num=126)).reshape(-1,1)
# xx_2 = (np.linspace(3,28,num=126)).reshape(-1,1)
# x_values = np.concatenate((xx_1, xx_2), axis=1)
# x_train = np.array(x_values, dtype=np.float32)
# yy_1 = (np.linspace(2,27,num=126)).reshape(-1,1)
# yy_2 = (np.linspace(3,28,num=126)).reshape(-1,1)
# y_values = np.concatenate((yy_1, yy_2), axis=1)
# y_train = np.array(y_values, dtype=np.float32)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

inputDim = 16  # takes variable 'x'
outputDim = 4  # takes variable 'y'
learningRate = 0.00001
epochs = 200000
hidden_size = 400
model = ActorCritic(inputDim, outputDim, hidden_size).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
abc = state_finder()
y_train = np.zeros((10, 4), dtype=np.float32)
for epoch in range(epochs):
    # Converting inputs and labels to Variable
    x_train = np.array(np.random.uniform(-0.1, 0.1, (10, 16)), dtype=np.float32)
    for i in range(x_train.shape[0]):
        y_train[i] = np.array(abc.get_action(x_train[i]), dtype=np.float32).reshape(4)
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(x_train).cuda())
        labels = Variable(torch.from_numpy(y_train).cuda())
    else:
        inputs = Variable(torch.from_numpy(x_train))
        labels = Variable(torch.from_numpy(y_train))

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs, value = model(inputs)
    # get loss for the predicted output
    loss = criterion(outputs, labels)
    if epoch % 500 == 0:
        print(loss)
    # get gradients w.r.t to parameters
    loss.backward()

    # update parameters
    optimizer.step()
    if epoch % 500 == 0:
        print("epoch {}, loss {}".format(epoch, loss.item()))

# with torch.no_grad():  # we don't need gradients in the testing phase
#     if torch.cuda.is_available():
#         predicted = model(Variable(torch.from_numpy(x_train).cuda())).cpu().data.numpy()
#     else:
#         predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
#     print(predicted)

# plt.clf()
# plt.plot(x_train, y_train, "go", label="True data", alpha=0.5)
# plt.plot(x_train, predicted, "--", label="Predictions", alpha=0.5)
# plt.legend(loc="best")
# plt.show()


torch.save(model.state_dict(), "model.pt")
# model1 = PolicyNetwork(1, 1, 400, init_w=3e-3, device='cuda')
# # model1.load_state_dict(torch.load('model.pt'))
# model1.eval()
# model1(inputs)
