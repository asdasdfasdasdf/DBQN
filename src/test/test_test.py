from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt

from dataloader.dataloader import dataloader_de, dataloader_psd
from rl.train import *
import pytest

matplotlib.use('TkAgg')


class TestTest():

    def test_DE_N170(self):
        X_test, Y = dataloader_de("N170", 16, 21)
        _, EEG_channel, _, band = X_test.shape

        model_test = DBQN(EEG_channel, band, action_nums)
        model_test.to(device)

        model_test.load_state_dict(torch.load("./m/critic_DE_N170.pth"))

        pred = predict(model_test, X_test)
        ac = accuracy_score(Y, pred.cpu())
        print(Y)
        print(pred)
        print("accuracy score:", ac)

    # def test_PSD_N170(self):
    #     X_test, Y = dataloader_psd("N170", 16, 21)
    #     _, EEG_channel, _, band = X_test.shape
    #
    #     model_test = DBQN(EEG_channel, band, action_nums)
    #     model_test.to(device)
    #
    #     model_test.load_state_dict(torch.load("./m/critic_PSD_N170.pth"))
    #
    #     pred = predict(model_test, X_test)
    #     ac = accuracy_score(Y, pred.cpu())
    #     print(Y)
    #     print(pred)
    #     print("accuracy score:", ac)
    #
    # def test_DE_N270(self):
    #     X_test, Y = dataloader_de("N270", 16, 21)
    #     _, EEG_channel, _, band = X_test.shape
    #
    #     model_test = DBQN(EEG_channel, band, action_nums)
    #     model_test.to(device)
    #
    #     model_test.load_state_dict(torch.load("./m/critic_DE_N270.pth"))
    #
    #     pred = predict(model_test, X_test)
    #     ac = accuracy_score(Y, pred.cpu())
    #     print(Y)
    #     print(pred)
    #     print("accuracy score:", ac)
    #
    # def test_PSD_N270(self):
    #     X_test, Y = dataloader_psd("N270", 16, 21)
    #     _, EEG_channel, _, band = X_test.shape
    #
    #     model_test = DBQN(EEG_channel, band, action_nums)
    #     model_test.to(device)
    #
    #     model_test.load_state_dict(torch.load("./m/critic_PSD_N270.pth"))
    #
    #     pred = predict(model_test, X_test)
    #     ac = accuracy_score(Y, pred.cpu())
    #     print(Y)
    #     print(pred)
    #     print("accuracy score:", ac)


if __name__ == '__main__':
    pytest.main(['-vs', 'test_test.py'])