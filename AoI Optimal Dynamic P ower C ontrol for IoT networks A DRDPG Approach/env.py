import numpy as np
#修改增加能量收集场景

class env():
    def __init__(self):
        # 论文模型参数
        self.node_num = 5
        self.important_weight = 1 / self.node_num
        self.distance = np.zeros(self.node_num)
        self.bandwidth = 1e5
        self.maxAoI = 100
        self.episode = 300
        self.epoch = 1000
        self.count= 0
        for i in range(1, self.node_num+1):
            self.distance[i - 1] = min(2 + round(18 / (self.node_num - 1))* (i - 1),20)

        self.R = np.zeros(self.node_num)

        for i in range(1, self.node_num+1):
            self.R[i - 1] = max(0.2 -((0.2 - 0.02) / (self.node_num - 1)) * (i - 1) ,0.02)*1e6

        self.sigma_2 = 10 ** (-9.4)#单位为w
        self.channel_gain = []
        # 状态空间 AoI

        self.source_AoI = np.zeros(self.node_num)
        self.bs_AoI = np.zeros(self.node_num)
        self.po_AoI = np.zeros(self.node_num)

        # 信息生成概率
        self.data_generate_prob = 0.2

    def generate_gain(self):
        self.channel_gain=[]
        r = np.random.normal(0, 1, self.node_num)
        i = np.random.normal(0, 1, self.node_num)
        r = 1 / pow(2, 0.5) * r
        i = 1 / pow(2, 0.5) * i
        for a, b in zip(r, i):
            c = complex(a, b)
            self.channel_gain.append(abs(c))
        self.channel_gain = np.array(self.channel_gain)
        self.channel_gain = self.channel_gain * self.distance ** -3

    def generate_reward_and_update_AoI(self, action):
        # 更新源节点AoI
        data = np.random.binomial(1, 0.2, self.node_num)  # 生成信息

        for i, j in zip(range(self.node_num), data):
            self.source_AoI[i] = min(self.source_AoI[i] * (1-j) + 1, self.maxAoI)

        # 计算SINR
        total_rate = (action * self.channel_gain).sum()
        ifence = []
        for i in range(self.node_num):
            temp = total_rate - action[i] * self.channel_gain[i]
            ifence.append(temp)
        ifence = np.array(ifence)
        R = self.bandwidth * np.log2(1 + action * self.channel_gain / (ifence + self.sigma_2))
        judge = R >= self.R
        # 部分可观测源节点AoI,没有发射成功就保持不变
        # 更新基站AoI
        for i, j in zip(range(self.node_num), judge):
            if j == 1 and data[i]!=1:
                self.bs_AoI[i] = min(self.source_AoI[i], self.maxAoI)
                self.po_AoI[i] = self.bs_AoI[i]
            else:
                self.bs_AoI[i] = min(self.bs_AoI[i] + 1, self.maxAoI)



        # reward
        reward = -(self.bs_AoI * self.important_weight).sum()
        return reward

    def reset(self):
        self.count=1
        self.po_AoI = np.zeros(self.node_num)
        self.source_AoI=np.zeros(self.node_num)
        self.bs_AoI = np.zeros(self.node_num)
        state = np.append(self.po_AoI, self.bs_AoI).astype('float32')
        state=state/100
        return state

    def step(self, action):
        self.generate_gain()
        reward = self.generate_reward_and_update_AoI(action)
        next_state = np.append(self.po_AoI, self.bs_AoI).astype('float32')
        next_state=next_state/100
        if self.count==self.epoch:
            done=1
        else:
            done=0
            self.count+=1
        return next_state,reward,done


