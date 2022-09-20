import numpy as np
import torch
import torch.nn as nn
import random

class config():
    def __init__(self):
        #网络设置参数
        self.node_num=5
        self.actor_learning_rate=0.0005    #0.0004
        self.critic_learning_rate=0.0005   #0.0008   不做初始化
        self.episode = 300
        self.epoch = 1000
        self.buffer_size=4e5
        self.seqlen=20
        self.batch_size=10
        self.net_update_rate=1e-3
        self.max_AOI=100
        self.actor_input_size=self.node_num*2
        self.actor_output_size=self.node_num
        self.critic_input_size=self.node_num*3
        self.critic_output_size=1
        self.hidden_size=128
        self.dicount_rate=0.99
        self.device = torch.device('cuda:0')

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

def init_weight(lstm):
    for name, param in lstm.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.orthogonal_(param)


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.config=config()
        self.linear1=nn.Linear(self.config.actor_input_size,self.config.hidden_size)
        self.lstm=nn.LSTM(self.config.hidden_size,self.config.hidden_size,batch_first=True)
        self.linear2=nn.Linear(self.config.hidden_size,self.config.actor_output_size)
        #orthogonal_init(self.linear1)
        #init_weight(self.lstm)
        #orthogonal_init(self.linear2)
        self.f1=nn.ReLU()
        self.f2=nn.Tanh()

    def train_forward(self,x):#x(batch,seqlen,input)
        h_0=torch.zeros((1,self.config.batch_size,self.config.hidden_size)).to(self.config.device)
        c_0=torch.zeros((1,self.config.batch_size,self.config.hidden_size)).to(self.config.device)
        x=self.f1(self.linear1(x))#linear层输入任意维度，输出相同的维度
        x,(h_n,c_n)=self.lstm(x,(h_0,c_0))
        x=(self.f2(self.linear2(x))+1)*0.1/2
        #x=self.f2(self.linear2(x))
        #x=x.view(-1,self.config.actor_output_size)#x变为2维，不同回合的连续状态接在一起
        return x
    def interact_forward(self,x,h,c):#x(seqlen:1,input) 也可以设置成三维的batch=1
        x=self.f1(self.linear1(x))
        x,(h_n,c_n)=self.lstm(x,(h,c))
        x=(self.f2(self.linear2(x))+1)*0.1/2
        #x=self.f2(self.linear2(x))
        x=x.flatten()#展平成一维
        return x,h_n,c_n

    def init_h0_c0(self):
        return torch.zeros(1,self.config.hidden_size).to(self.config.device),\
               torch.zeros(1,self.config.hidden_size).to(self.config.device)

class Critrc(nn.Module):
    def __init__(self):
        super(Critrc, self).__init__()
        self.config=config()
        self.linear1=nn.Linear(self.config.critic_input_size,self.config.hidden_size)
        self.lstm=nn.LSTM(self.config.hidden_size,self.config.hidden_size,batch_first=True)
        self.linear2=nn.Linear(self.config.hidden_size,self.config.critic_output_size)
        #orthogonal_init(self.linear1)
        #init_weight(self.lstm)
        #orthogonal_init(self.linear2)
        self.f1=nn.ReLU()

    def forward(self,x): #x(batch,seqlen,inputsize)
        h_0=torch.zeros(1,self.config.batch_size,self.config.hidden_size).to(self.config.device)
        c_0=torch.zeros(1,self.config.batch_size,self.config.hidden_size).to(self.config.device)
        x=self.f1(self.linear1(x))
        x,(h_n,c_n)=self.lstm(x,(h_0,c_0))
        x=self.linear2(x)
        #x=x.view(-1,self.config.critic_output_size)#变成二维
        return x



class DRDPG():
    def __init__(self,env):
        self.config=config()
        self.update_actor=Actor().to(self.config.device)
        self.target_actor=Actor().to(self.config.device)
        self.update_critic=Critrc().to(self.config.device)
        self.target_critic=Critrc().to(self.config.device)
        # 复制参数到目标网络
        for target_param, param in zip(self.target_critic.parameters(), self.update_critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.update_actor.parameters()):
            target_param.data.copy_(param.data)
        #定义优化器，loss function
        self.loss_fun=nn.MSELoss().to(self.config.device)
        self.actor_optim=torch.optim.Adam(self.update_actor.parameters(),lr=self.config.actor_learning_rate)
        self.critic_optim=torch.optim.Adam(self.update_critic.parameters(),lr=self.config.critic_learning_rate)

        self.temp_buffer=[]
        self.buffer=[]

        self.env=env

        self.num=0
        self.num1=0
    def add_noise(self,action):
        #action=(action+1)*0.1/2
        noise=np.random.normal(0,0.1,self.config.node_num)
        noise=np.clip(noise,-0.001,0.005)#DDPG先要剪裁噪声
        noise_action=np.clip(action+noise,0.0,0.1)
        return noise_action

    def push_temp_buff(self,experience):
        self.temp_buffer.append(experience)

    def clean_temp_buff(self):
        self.temp_buffer=[]

    def tranfer_to_buffer(self):
        #get_len=np.array(self.buffer,dtype=object)
        #gen_len=get_len.reshape(-1,5)#是否能转化成二维
        if len(self.buffer)==self.config.buffer_size/self.config.epoch:
            self.buffer.pop(0)
        else:
            self.buffer.append(self.temp_buffer)

    def sample_and_disposal(self):
        epoisde=random.sample(self.buffer,self.config.batch_size)
        #epoisde=np.array(epoisde)
        #获得batch,(batch_size,seqlen,h)
        state=[]
        action=[]
        reward=[]
        next_state=[]
        done=[]
        for i in epoisde:
            idx=np.random.randint(980,size=1)
            #print(type(idx))
            train_data=i[idx[0]:idx[0]+20]#索引是int，不能是narray
            state_=[  k[0]  for k in train_data]
            action_=[k[1] for k in train_data]
            reward_=[ [k[2]] for k in train_data ]
            next_state_=[k[3] for k in train_data]
            done_=[[k[4]] for k in train_data]
            state.append(state_)
            action.append(action_)
            reward.append(reward_)
            next_state.append(next_state_)
            done.append(done_)
        #转为tensor
        state=np.array(state)
        action=np.array(action).astype('float32')
        reward=np.array(reward).astype('float32')
        next_state=np.array(next_state)
        done=np.array(done).astype('float32')

        state=torch.from_numpy(state).to(self.config.device)
        action=torch.from_numpy(action).to(self.config.device)
        reward=torch.from_numpy(reward).to(self.config.device)
        next_state=torch.from_numpy(next_state).to(self.config.device)
        done=torch.from_numpy(done).to(self.config.device)

        return state,action,reward,next_state,done

    def train(self,epoisde):
        state=self.env.reset()
        rewards=0
        h,c=self.update_actor.init_h0_c0()
        while True:
            state_=torch.from_numpy(state).unsqueeze(0).to(self.config.device)
            action,h,c=self.update_actor.interact_forward(state_,h,c)
            action=action.detach().cpu().numpy()
            action=self.add_noise(action)
            next_state,reward,done=self.env.step(action)
            self.push_temp_buff([state,action,reward,next_state,done])
            rewards+=reward
            if epoisde>=12:
                self.update(done)
            if done:
                break
            else:
                state=next_state
        self.tranfer_to_buffer()
        self.clean_temp_buff()
        return rewards/self.config.epoch


    def update(self,x):
        state, noise_action, reward, next_state, done=self.sample_and_disposal()
        self.num += 1
        self.num1 += 1
        self.critic_optim.zero_grad()
        critic_state_input = torch.cat((state, noise_action), 2)  # 相同数组拼接函数，dim=0表示行拼接，1表示列拼接
        next_action = self.target_actor.train_forward(next_state)
        critic_nextstate_input = torch.cat((next_state, next_action), 2)
        q_state_value = self.update_critic(critic_state_input)
        q_nextstate_value = self.target_critic(critic_nextstate_input)
        target_q = reward + (1 - done) * self.config.dicount_rate * q_nextstate_value
        loss = self.loss_fun(target_q, q_state_value)
        loss.backward()
        # print(loss.grad)
        self.critic_optim.step()
        # 更新策略网络
        if self.num1 == 2:
            self.actor_optim.zero_grad()
            true_action=self.update_actor.train_forward(state)
            true_q_input = torch.cat((state, true_action), 2)
            if x==1:
                print(true_action.data[4,1])
                print(state.data[4,1])
                print(reward.data[4,1])
            true_q_value = self.update_critic(true_q_input).mean()
            true_q_value = -true_q_value
            true_q_value.backward()
            # print(self.policy_net.linear3.weight.grad)
            self.actor_optim.step()
            self.num1 = 0
        # 对网络进行软更新
        if self.num == 1:
            for target_param, param in zip(self.target_critic.parameters(), self.update_critic.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.config.net_update_rate) +
                    param.data * self.config.net_update_rate
                )
            for target_param, param in zip(self.target_actor.parameters(), self.update_actor.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.config.net_update_rate) +
                    param.data * self.config.net_update_rate
                )
            self.num = 0

    def evaluate(self):
        state = self.env.reset()
        rewards = 0
        h, c = self.update_actor.init_h0_c0()

        while True:
            state_ = torch.from_numpy(state).unsqueeze(0).to(self.config.device)
            action, h, c = self.update_actor.interact_forward(state_, h, c)
            action = action.detach().cpu().numpy()
            next_state, reward, done = self.env.step(action)
            rewards += reward
            if done:
                break
            else:
                state = next_state
        return rewards / self.config.epoch






















        










