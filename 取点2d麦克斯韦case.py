import numpy as np
import random
np.random.seed(10)
random.seed(10)
import torch


def target_density_function(vx, vy):
    return 1 / np.pi * np.exp(-vx ** 2 - vy ** 2) * (vx ** 2 + vy ** 2)

def proposal_distribution(vx, vy):
    return np.random.multivariate_normal(mean=np.array([vx, vy]), cov=np.eye(2))

def proposal_density_function(vx, vy, mean1, mean2):
    return 1 / (2 * np.pi) * np.exp((-(vx - mean1) ** 2 - (vy - mean2) ** 2) / 2)

def metropolis_hastings(initial_data, n_samples):
    current_data_x = initial_data[0]
    current_data_y = initial_data[1]
    accepted_samples = []
    for i in range(n_samples*5+5000):
        proposed_data = proposal_distribution(current_data_x, current_data_y).ravel()
        A = target_density_function(proposed_data[0], proposed_data[1]) * proposal_density_function(vx=current_data_x,vy=current_data_y,
            mean1=proposed_data[0],mean2=proposed_data[1])
        B = target_density_function(current_data_x, current_data_y) * proposal_density_function(vx=proposed_data[0],
            vy=proposed_data[1],mean1=current_data_x, mean2=current_data_y)
        r = np.minimum(A / B, 1)
        u = np.random.rand()
        if u <= r:
            current_data_x = proposed_data[0]
            current_data_y = proposed_data[1]
            accepted_samples.append(proposed_data)
        else:
            accepted_samples.append(np.array([current_data_x, current_data_y]))
        if i % 10000 == 0:
            print(i)

    return np.array(accepted_samples)[5000::5,]

a=metropolis_hastings(initial_data=[-1.414,0],n_samples=100000)
#print(np.mean(np.linalg.norm(a,axis=1)**2))
np.save(f'100000_points_2d-Maxwell', a)


def BKW_np(v, t=0):
    K = 1 - np.exp(-t / 8) / 2
    u = 1 / (2 * torch.pi * K) * np.exp(-(v ** 2).sum(axis=1) / (2 * K)) * (
                (2 * K - 1) / K + (1 - K) / (2 * K * K) * (v ** 2).sum(axis=1))
    return u.reshape(-1, 1)

class Config(object):
    dim = 2
    gamma = 0
    lamda = 1 / 16
    # 网络参数
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_sample = 100000  # 每次迭代模拟粒子数
    n_0 = num_sample ** (1 / dim)  # 用于算eps

    truncation = 4  # 因为是有限传播速度，可以做空间截断，避免不必要的计算
    total_time = 200  # pme方程的时间，不能太大
    num_time_interval = 2000  # 时间离散数量，每个离散点用一个网络来计算，不能太多

config = Config()
device = config.device

def initial_sample_analy(number, truncation=config.truncation, t=0):  # 去掉强制对称的分布
    # number = number // 2
    samplenumber = number * 10
    output = np.zeros([1, config.dim])
    while output.size / config.dim < number + 1:
        x = 2.0 * truncation * torch.rand((samplenumber, config.dim)).numpy() - truncation
        y_max = 0.5  # 最大值也要相应修改
        y = np.random.uniform(0, y_max, samplenumber)
        # x2 = ((x**2).sum(axis=1))
        u = BKW_np(x, t).reshape(-1)  # 这里更改初始的密度函数
        accept = x[np.where(y < u)]
        output = np.append(output, accept, axis=0)
        print(output.size / config.dim)
    output = output[1:number + 1]
    return output

#b = initial_sample_analy(100000)
#print(np.mean(np.linalg.norm(b, axis=1) ** 2))
#np.save(f'100000_points_2d-Maxwell', b)


