from dataloader.dataloader import dataloader_de, dataloader_psd
from rl.environment import MyEnvironment
import torch
import pytest

class TestData():
    def test_dataloader_de(self):
        X_train, Y = dataloader_de("N170",1,21)
        # 验证最终形状
        print()
        print("________________________________________________________________________")
        print("X_train shape:", X_train.shape) #[40, 43, 2, 5]
        print("X_test type:", X_train.type())

        print("Y shape:", Y.shape)
        print("样本标签示例:", Y[:10])

    def test_de_env(self):
        X_train, Y = dataloader_de("N170", 1, 21)
        # [40, 43, 2, 5]
        env = MyEnvironment(X_train,Y)
        state = env.nex()
        next_state, r= env.step(1)
        new_state = env.nex()
        print()
        print("________________________________________________________________________")
        print(state.shape)
        print(next_state.shape)
        print(new_state.shape)

    def test_dataloader_psd(self):
        X_train, Y = dataloader_psd("N170",1,21)
        # 验证最终形状
        print()
        print("________________________________________________________________________")
        print("X_train shape:", X_train.shape) #[40, 1, 43, 2, 5]
        print("X_test type:", X_train.type())

        print("Y shape:", Y.shape)
        print("样本标签示例:", Y[:10])

    def test_psd_env(self):
        X_train, Y = dataloader_psd("N170", 1, 21)
        env = MyEnvironment(X_train,Y)
        state = env.nex()
        next_state, r= env.step(1)
        new_state = env.nex()
        print()
        print("________________________________________________________________________")
        print(state.shape)
        print(next_state.shape)
        print(new_state.shape)



if __name__ == '__main__':
	pytest.main(['-vs','test_data.py'])
