import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
'''
def get_data():
    获取数据
    
    basecond = np.array([[18, 20, 19, 18, 13, 4, 1],[20, 17, 12, 9, 3, 0, 0],[20, 20, 20, 12, 5, 3, 0]])
    cond1 = np.array([[18, 19, 18, 19, 20, 15, 14],[19, 20, 18, 16, 20, 15, 9],[19, 20, 20, 20, 17, 10, 0]])
    cond2 = np.array([[20, 20, 20, 20, 19, 17, 4],[20, 20, 20, 20, 20, 19, 7],[19, 20, 20, 19, 19, 15, 2]])
    cond3 = np.array([[20, 20, 20, 20, 19, 17, 12],[18, 20, 19, 18, 13, 4, 1], [20, 19, 18, 17, 13, 2, 0]])
    return basecond, cond1, cond2, cond3

data = get_data()
label = ['algo1', 'algo2', 'algo3', 'algo4']
df=[]
for i in range(len(data)):#加入dataframe到列表
    df.append(pd.DataFrame(data[i]).melt(var_name='episode',value_name='reward'))#melt自定义变量和值的名称
    df[i]['algo']= label[i]
df=pd.concat(df,ignore_index=True) # 合并 df为多个dataframe的列表，返回对象为dataframe,ignore_idex=True 重新排序索引
#df.to_csv('D:\pythonProject\AoI Optimal Dynamic P ower C ontrol for IoT networks A DRDPG Approach\data.csv')
print(df)
sns.lineplot(x="episode", y="reward", hue="seed", style="seed",data=df)
plt.title("average_reward")
plt.show()
'''
sns.set()
reward=pd.read_csv('reward.csv',usecols=[1,2])#usecols =[1,2],去出1,2列
print(reward)
reward.iloc[:,[1]]+=1#dataframe的切片，取出第一列
sns.lineplot(x='episode',y='average_reward',data=reward)
#sns.lineplot(x=range(800),y=Reward)
plt.title('average_reward')#图的标题
plt.xlim((0,800))#设置x轴的范围
plt.show()



