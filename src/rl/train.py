from tqdm import tqdm
from rl.environment import MyEnvironment
from rl.DBQN import *

def train(X_train, Y_train, name):

    #动作空间的大小
    env = MyEnvironment(X_train, Y_train)

    #state.shape = (size,EEG_channel,band)
    state = env.reset()

    _, EEG_channel, band= state.shape

    actor_model = DBQN(EEG_channel, band, action_nums, device)  # 用于决策
    critic_model = DBQN(EEG_channel, band, action_nums, device)  # 用于训练

    actor_model.to(device)
    critic_model.to(device)

    # 优化器选为adam
    optimizer = torch.optim.Adam(critic_model.parameters())

    memory.clear()
    reward_rec = [] #记录每epoch的总奖励
    pre_remember(env,pre_train_num)

    print()
    pbar = tqdm(range(1, epoches + 1))

    for epoch in pbar:
        total_rewards = 0
        for step in range(forward):
            # 对每个状态使用epsilon_greedy选择
            action, q = epsilon_greedy(env, state, epoch, actor_model, ep_min, ep_max, epislon_total)
            eps = epsilon_calc(epoch,ep_min, ep_max, esp_total=epislon_total)
            # play
            next_state, reward = env.step(action)
            # 加入到经验记忆中
            remember(state, action, q, reward, next_state)
            # 从记忆中采样回放，保证iid。实际上这个任务中这一步不是必须的。
            loss = replay(critic_model, actor_model,optimizer)
            total_rewards += reward
            state = env.nex()
            if ((epoch - 1) * forward + step + 1) % every_copy_step == 0:
                copy_critic_to_actor(critic_model, actor_model)
        reward_rec.append(total_rewards)
        pbar.set_description(
            'R:{} L:{:.4f} P:{:.3f}'.format(total_rewards, loss, eps))

    torch.save(critic_model.state_dict(),name)

    return reward_rec