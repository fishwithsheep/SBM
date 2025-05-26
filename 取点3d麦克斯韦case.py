import numpy as np
import random
np.random.seed(10)
random.seed(10)
import torch

def target_density_function(vx,vy,vz):
    K=0.6
    z=vx**2+vy**2+vz**2
    return (1/(2*np.pi*K)**1.5*np.exp(-z/(2*K))*((1-K)/(2*K**2)*z)).reshape([-1,1])

def proposal_distribution(vx,vy,vz):
    return np.random.multivariate_normal(mean=np.array([vx,vy,vz]),cov=np.eye(3)*0.6)

def proposal_density_function(vx,vy,vz,mean1,mean2,mean3):
    return 1/(1.2*np.pi)**1.5*np.exp((-(vx-mean1)**2-(vy-mean2)**2-(vz-mean3)**2)/1.2)

def metropolis_hastings(initial_data, n_samples):
    current_data_x = initial_data[0]
    current_data_y=initial_data[1]
    current_data_z = initial_data[2]
    accepted_samples = []
    for i in range(n_samples*5+5000):
        proposed_data = proposal_distribution(current_data_x,current_data_y,current_data_z).reshape([-1])
        A=target_density_function(proposed_data[0],proposed_data[1],proposed_data[2])*proposal_density_function(vx=current_data_x,
        vy=current_data_y,vz=current_data_z,mean1=proposed_data[0],mean2=proposed_data[1],mean3=proposed_data[2])
        B=target_density_function(current_data_x,current_data_y,current_data_z)*proposal_density_function(vx=proposed_data[0],
        vy=proposed_data[1],vz=proposed_data[2],mean1=current_data_x,mean2=current_data_y,mean3=current_data_z)
        r=np.minimum(A/B,1)
        u=np.random.random()
        if u<=r:
            accepted_samples.append(proposed_data)
            current_data_x=proposed_data[0]
            current_data_y=proposed_data[1]
            current_data_z=proposed_data[2]
        else:
            accepted_samples.append(np.array([current_data_x, current_data_y,current_data_z]))
        if i %10000==0:
            print(i)

    return np.array(accepted_samples)[5000::5,]

a=metropolis_hastings(initial_data=[1,0,0],n_samples=500000)
#print(np.mean(np.linalg.norm(a,axis=1)**2))
np.save(f'500000_points_3d-Maxwell',a)


def BKW_np(v,t=5.4977):
    K=1-np.exp(-t/6)
    u = 1/(2*torch.pi*K)**1.5*np.exp(-(v**2).sum(axis=1)/(2*K))*((5*K-3)/(2*K)+(1-K)/(2*K*K)*(v**2).sum(axis=1))
    return u.reshape(-1,1)

class Config(object):
    dim = 3
    gamma = 0
    lamda = 1/24
    # 网络参数
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_sample = 100000 #每次迭代模拟粒子数
    n_0 = num_sample**(1/dim)    #用于算eps

    truncation = 4          #因为是有限传播速度，可以做空间截断，避免不必要的计算
    total_time = 200        #pme方程的时间，不能太大
    num_time_interval = 2000 #时间离散数量，每个离散点用一个网络来计算，不能太多

config=Config()
device = config.device

def initial_sample_analy(number,truncation=config.truncation,t=5.4977):   #去掉强制对称的分布
    #number = number // 2
    samplenumber = number * 10
    output = np.zeros([1,config.dim])
    while output.size/config.dim < number+1:
        x = 2.0 * truncation * torch.rand((samplenumber, config.dim)).numpy() - truncation
        y_max = 0.5                                          #最大值也要相应修改
        y = np.random.uniform(0,y_max,samplenumber)
        #x2 = ((x**2).sum(axis=1))
        u = BKW_np(x,t).reshape(-1)                             #这里更改初始的密度函数
        accept = x[np.where(y<u)]
        output = np.append(output,accept,axis=0)
    output = output[1:number+1]
    return output
#b=initial_sample_analy(number=500000)
#print(np.mean(np.linalg.norm(b,axis=1)**2))
#np.save(f'500000_points_3d-Maxwell',b)
