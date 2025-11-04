import torch
from torch import nn

from collections import deque
import numpy as np
import random

from rl.layer import SNN

memory = deque(maxlen=32)
pre_train_num = 4
replay_size = 8

gamma = 0.8
alpha = 0.15

epoches = 30
forward = 50
every_copy_step = 32

ep_min = 0.1
ep_max = 1
epislon_total = 30

action_nums = 2
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
loss_fn = torch.nn.MSELoss().to(device)
seed = 42

class DBQN(nn.Module):
    def __init__(self, input_channel, input_band, actions_num, device):
        super(DBQN, self).__init__()
        self.device = device

        self.actions_num = actions_num

        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channel, input_channel*4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(input_channel*4, input_channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.local_snn = SNN(input_band * input_channel, 3 * input_band * input_channel, self.device, input_channel, input_band)

        # 动态计算全连接层输入尺寸
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channel, input_band)
            dummy_output = self.conv_layers(dummy_input)
            dummy_output = dummy_output.to(self.device)
            dummy_output1,dummy_output2 = self.local_snn(dummy_output)
            x = torch.cat((dummy_output1, dummy_output2), dim=2)
            self.flattened_size = x.view(1, -1).size(1)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.actions_num)
        )

    def forward(self, eeg_input, actions_input):
        eeg_input = eeg_input.to(self.device)
        actions_onehot_input = torch.nn.functional.one_hot(actions_input, self.actions_num)
        actions_onehot_input = actions_onehot_input.to(self.device)

        #特征提取
        x = self.conv_layers(eeg_input)
        x = x.detach()
        snn_x1, snn_x2 = self.local_snn(x)      # snn_x1:[batch_size,43,5*2] snn_x2:[batch_size,43,5*3]
        x = torch.cat((snn_x1,snn_x2),dim=2)    # x:[batch_size,43,5*5]

        x = x.view(x.size(0), -1)
        q_values = self.fc(x)

        # 动作选择处理
        q_value = torch.sum(q_values * actions_onehot_input, dim=1, keepdim=False)
        return q_value

    def get_q_values(self, eeg_input):
        eeg_input = eeg_input.to(self.device)

        x = self.conv_layers(eeg_input)
        x = x.detach()
        snn_x1 , snn_x2 = self.local_snn(x)     # snn_x1:[batch_size,43,5*2] snn_x2:[batch_size,43,5*3]
        x = torch.cat((snn_x1,snn_x2),dim=2)

        x = x.view(x.size(0), -1)
        q_values = self.fc(x)
        return q_values

def copy_critic_to_actor(critic_model, actor_model):
    actor_model.load_state_dict(critic_model.state_dict())

#epsilon_greedy策略实现
# 用于动作选择与动作对应的q值获取
def epsilon_calc(step, ep_min=0.01,ep_max=1,esp_total = 1000):
    return max(ep_min, ep_max -(ep_max - ep_min)*step/esp_total )

def epsilon_greedy(env, state, step,actor_model, ep_min=0.01,ep_max = 1, ep_total=1000):
    epsilon = epsilon_calc(step, ep_min, ep_max,ep_total)
    with torch.no_grad():
        qvalues = actor_model.get_q_values(state)
    i = env.sample_actions()

    if np.random.rand()<epsilon:
        return i,qvalues[0][i]
    return torch.argmax(qvalues,dim=1), torch.max(qvalues,dim=1)[0]



def remember(state, action, action_q, reward, next_state):
    memory.append([state, action, action_q, reward, next_state])

def sample_ram(sample_num):
    return random.sample(memory, sample_num)

def pre_remember(env, pre_go=30):
    state = env.reset()
    for i in range(pre_go):
        rd_action = env.sample_actions()
        next_state, reward = env.step(rd_action)
        remember(state, rd_action, 0, reward, next_state)
        state = env.nex()


def replay(critic_model, actor_model,optimizer):
    if len(memory) < replay_size:
        return 0.0
    samples = sample_ram(replay_size)

    # 解包并转换数据
    states, actions, old_qs, rewards, next_states = zip(*samples)

    # (size, 1, EEG_channel, band)
    states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states])
    actions = torch.tensor(actions, dtype=torch.int64)
    old_qs = torch.tensor(old_qs, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    # (size, 1, EEG_channel, band)
    next_states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in next_states])

    # (size, EEG_channel, band)
    states = states[:,0,:,:]
    next_states = next_states[:,0,:,:]

    old_qs = old_qs.to(device)
    rewards = rewards.to(device)

    #训练流程
    critic_model.train()
    # 计算Q值（禁用梯度计算）
    with torch.no_grad():
        qvalues = actor_model.get_q_values(next_states)
        qs = torch.max(qvalues, dim=1)[0]
        # 计算Q估计值
        q_estimates = (1 - alpha) * old_qs + alpha * (rewards + gamma * qs)

    predicted_qs = critic_model(states, actions)
    loss = loss_fn(predicted_qs, q_estimates)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def predict(model,states):
   model.eval()
   # states (size, EEG_channel, time_step, band)
   with torch.no_grad():
       for time_step in range(states.shape[2]):
           time_slice = states[:, :, time_step, :]
           if time_step == 0:
               qvalues = model.get_q_values(time_slice)
           else:
               qvalues = qvalues + model.get_q_values(time_slice)
   return torch.argmax(qvalues, dim=1)
