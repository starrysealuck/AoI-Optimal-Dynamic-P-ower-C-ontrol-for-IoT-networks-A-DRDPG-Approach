import os
from DRDPG import DRDPG
from env import env
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import pandas as pd
def seed_fixed(seed):
    # 要固定这些随机种子结果才能相同
    # self.env.reset(seed=1029)  # 初始化环境时的随机种子
    # self.env.action_space.seed(1029)  # 动作采样的随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
if __name__ == '__main__':
    #seed_eval_reward=[]
    #for i in range(1,11):
    #seed=random.randint(1,1000)
    seed_fixed(10)  #657,10
    env=env()
    drdpg=DRDPG(env)
    #Reward=[]
    rewards=0
    eval_Reward=[]
    for i in range(env.episode):
        reward=drdpg.train(i+1)
        #print('第%d训练回合 平均奖励 %.4f:' % (i+1, rewards))
        eval_reward=drdpg.evaluate()
        if len(eval_Reward)==0:
            rewards=eval_reward
            eval_Reward.append(rewards)
        else:
            rewards=0.7*rewards+0.3*eval_reward
            eval_Reward.append(rewards)
        print(' 第%d训练回合 评估平均奖励: %.4f' %(i+1,rewards))
        #seed_eval_reward.append(eval_Reward)

    plt.plot(range(env.episode),eval_Reward)
    plt.xlabel('epoisde')
    plt.ylabel('average reward')
    plt.show()

    #seed_eval_reward=pd.DataFrame(seed_eval_reward).melt(var_name='episode',value_name='average_reward')
    #seed_eval_reward.to_excel()
