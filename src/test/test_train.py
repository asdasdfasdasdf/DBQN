from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
import torch

from dataloader.dataloader import dataloader_de, dataloader_psd
from rl.train import *
import pytest

matplotlib.use('TkAgg')

class TestTrain():
    # def test_model(self):
    #     X_train, Y_train = dataloader_de("N170", 1, 16)
    #     env = MyEnvironment(X_train,Y_train)
    #
    #     state = env.nex()
    #     _, EEG_channel, band = state.shape
    #
    #     model = DBQN(EEG_channel, band, action_nums,device)
    #     model.to(device)
    #
    #     action_0 = torch.tensor(0)
    #     action_1 = torch.tensor(1)
    #
    #     with torch.no_grad():
    #         o1 = model.get_q_values(state)
    #         o2 = model.forward(state, action_0)
    #         o3 = model(state, action_1)
    #
    #     print(o1)
    #     print(o2)
    #     print(o3)

    def test_DE_N170(self):

        X_train, Y_train = dataloader_de("N170", 1, 16)

        reward_rec = train(X_train, Y_train, "./m/critic_DEN170.pth")

        plt.plot(range(len(reward_rec)), reward_rec, c='b')
        plt.xlabel('iters')
        plt.ylabel('DE_N170 mean score')
        plt.show()

if __name__ == '__main__':
	pytest.main(['-vs','test_train.py'])