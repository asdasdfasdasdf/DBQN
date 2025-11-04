import torch
from torch.nn.modules.module import Module
import torch.nn as nn
import numpy as np



############################## SNN ##############################
# 往SNN上改
NEURON_VTH = 0.5
NEURON_CDECAY = 1 / 2
NEURON_VDECAY = 3 / 4
SPIKE_PSEUDO_GRAD_WINDOW = 0.5

class PseudoSpikeRect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_pseudo_grad = (abs(input - NEURON_VTH) < SPIKE_PSEUDO_GRAD_WINDOW)
        return grad_input * spike_pseudo_grad.float()


class ActorNetSpiking(nn.Module):
    """ Spiking Actor Network """
    def __init__(self, state_num, action_num, device, batch_window=5, hidden1=430):
        """

        :param state_num: number of states
        :param action_num: number of actions
        :param device: device used
        :param batch_window: window steps
        :param hidden1: hidden layer 1 dimension
        """
        super(ActorNetSpiking, self).__init__()
        self.state_num = state_num
        self.action_num = action_num
        self.device = device
        self.batch_window = batch_window
        self.hidden1 = hidden1
        self.pseudo_spike = PseudoSpikeRect.apply
        self.fc1 = nn.Linear(self.state_num, self.hidden1, bias=True)
        self.fc2 = nn.Linear(self.hidden1, self.action_num, bias=True)


    def neuron_model(self, syn_func, pre_layer_output, current, volt, spike):
        """
        Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param current: current of last step
        :param volt: voltage of last step
        :param spike: spike of last step
        :return: current, volt, spike
        """
        # if flag == 1 : volt = 0
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        volt = volt * NEURON_VDECAY * (1. - spike) + current
        spike = self.pseudo_spike(volt)
        return current, volt, spike

    def forward(self, x, batch_size):
        """

        :param x: state batch
        :param batch_size: size of batch
        :return: out
        """
        fc1_u = torch.zeros(batch_size, self.hidden1, device=self.device)
        fc1_v = torch.zeros(batch_size, self.hidden1, device=self.device)
        fc1_s = torch.zeros(batch_size, self.hidden1, device=self.device)
        fc2_u = torch.zeros(batch_size, self.action_num, device=self.device)
        fc2_v = torch.zeros(batch_size, self.action_num, device=self.device)
        fc2_s = torch.zeros(batch_size, self.action_num, device=self.device)
        fc1_sumspike = torch.zeros(batch_size, self.hidden1, device=self.device)
        fc2_sumspike = torch.zeros(batch_size, self.action_num, device=self.device)
        for step in range(self.batch_window):
            input_spike = x[:, :, step]
            fc1_u, fc1_v, fc1_s = self.neuron_model(self.fc1, input_spike, fc1_u, fc1_v, fc1_s)
            fc2_u, fc2_v, fc2_s = self.neuron_model(self.fc2, fc1_s, fc2_u, fc2_v, fc2_s)
            fc1_sumspike += fc1_s
            fc2_sumspike += fc2_s
        out1 = fc1_sumspike / self.batch_window    # out1->local_x1  out2->local_x2 为了和PGCN数据维度对齐
        out2 = fc2_sumspike / self.batch_window
        return out1, out2

class PoissonEncoder:
    def __init__(self, spike_state_num, batch_window):
        self.spike_state_num = spike_state_num
        self.batch_window = batch_window

        # SNN相关函数部分
    def state_2_state_spikes(self, spike_state_value, batch_size):  # batch_size = 8; batch_window = 5
        """
        Transform state to spikes of input neurons
        :param spike_state_value: state from environment transfer to firing rates of neurons
        :param batch_size: batch size
        :return: state_spikes
        """
        spike_state_value = spike_state_value.reshape((-1, self.spike_state_num, 1))
        state_spikes = np.random.rand(batch_size, self.spike_state_num, self.batch_window) < spike_state_value
        state_spikes = state_spikes.astype(float)
        return state_spikes

class SNN(Module):
    def __init__(self, in_features, out_features, device, input_channel, input_band):
        super(SNN, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.input_channel = input_channel
        self.input_band = input_band
        self.actor_net_spiking = ActorNetSpiking(self.in_features, self.out_features, self.device)

    def forward(self, input):
        # input(replay_size,43,5) -> input(replay_size,215) -> input(replay_size,215,5)
        # print('input.shape:', input.shape)
        # print('input:',input)
        batch_size = input.size(0)
        input_spread = input.view(batch_size, -1)   # 将 input_tensor 转换为形状为 (replay_size, 215)
        input_spread_np = input_spread.cpu().numpy()# 将 input_spread 转换为 numpy 数组以适应泊松编码器.首先将张量移动到 CPU,然后转换为 numpy 数组
        encoder = PoissonEncoder(spike_state_num= self.input_channel * self.input_band, batch_window=5)        # 创建泊松编码器实例
        batch_size = input_spread.size(0)
        state_spikes_np = encoder.state_2_state_spikes(input_spread_np, batch_size)                # 将 input_spread 输入泊松编码函数，得到形状为 (replay_size, 215, 5) 的张量 x
        state_spikes = torch.tensor(state_spikes_np, dtype=torch.float32).to(self.device)          # 将结果转换回 PyTorch 张量并转到cuda上

        output1, output2 = self.actor_net_spiking(state_spikes, batch_size)                        # output1(replay_size,43*5*2)   output2(replay_size,43*5*3)

        output_reshaped1 = output1.view(batch_size, self.input_channel, self.input_band * 2)    #output_reshaped1(replay_size,43,10)
        output_reshaped2 = output2.view(batch_size, self.input_channel, self.input_band * 3)    #output_reshaped2(replay_size,43,15)
        return output_reshaped1, output_reshaped2 # (replay_size, 43, out_features)


    def __repr__(self):
        return f"{self.__class__.__name__}: {str(self.in_features)} -> {str(self.out_features) }"
############################## SNN ##############################


