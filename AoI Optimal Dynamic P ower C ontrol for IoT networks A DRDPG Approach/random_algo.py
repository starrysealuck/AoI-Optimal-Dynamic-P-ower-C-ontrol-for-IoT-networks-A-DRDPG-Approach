from env import env
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
class random_algo():
    def __init__(self,env):
        self.env=env
    def random_train(self):
        state=self.env.reset()
        rewards=0
        while True:
            action=np.random.uniform(low=0,high=0.1,size=self.env.node_num)
            next_state,reward,done=self.env.step(action)
            rewards+=reward
            if done:
                break
            else:
                state=next_state
        print(rewards/self.env.epoch)
        return rewards/self.env.epoch


if __name__ == '__main__':
    Reward=[]
    env=env()
    random_=random_algo(env)
    rewards=0
    for i in range(env.episode):
        reward=random_.random_train()
        if len(Reward)==0:
            rewards=reward
            Reward.append(rewards)
        else:
            rewards=0.5*rewards+0.5*reward
            Reward.append(rewards)
    sns.set()
    reward = pd.read_csv('reward.csv', usecols=[1, 2])
    reward.iloc[:, [0]] += 1
    sns.lineplot(x='episode', y='average_reward', data=reward)
    sns.lineplot(x=range(800), y=Reward)
    plt.title('average_reward')
    plt.xlim((0, 800))
    plt.show()
