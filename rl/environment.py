import random
from rl.DQN import action_nums

class MyEnvironment(object):
    def __init__(self, x, y):
        self.train_X = x
        self.train_Y = y
        self.current_index = self._sample_index()
        self.action_space = action_nums

    def reset(self):
        obs= self.nex()
        return obs

    '''
    action: category, -1 : start and no reward
    return: next_state, reward
    '''
    def nex(self):

        # train_X   (size,EEG_channel,time_step,band)
        # return    (1,  ,EEG_channel          ,band)
        return self.train_X[self.current_index,:,0,:].unsqueeze(0)

    def step(self, action):
        r = self.reward(action)
        temp = self.current_index
        self.current_index = self._sample_index()
        return self.train_X[temp,:,1,:].unsqueeze(0), r

    def reward(self, action):
        c = self.train_Y[self.current_index]
        return 1 if c == action else -1

    def sample_actions(self):
        return random.randint(0, self.action_space - 1)

    def _sample_index(self):
        return random.randint(0, len(self.train_Y) - 1)