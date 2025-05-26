import matplotlib.pyplot as plt
import numpy as np
import random
np.random.seed(10)
random.seed(10)
import time

def mollifier(d,x,epsilon):
    'x是(n,d)矩阵，输出(n,1)矩阵'
    return (2*np.pi*epsilon)**(-d/2)*np.exp(-np.linalg.norm(x,axis=-1)**2/(2*epsilon)).reshape(-1,1)

def gradient_mollifier(d,x,epsilon):
    'x是(n,d)矩阵，输出(n,d)矩阵'
    return (2*np.pi*epsilon)**(-d/2)*np.multiply(np.exp(-np.linalg.norm(x,axis=-1)**2/(2*epsilon)).reshape(-1,1),(-x/epsilon))

def collision_kernel_A(Lambda,d,x,gamma):
    'x是(1,d)矩阵，输出(d,d)矩阵'
    a=np.linalg.norm(x,axis=-1)**2*np.eye(d)
    b = np.matmul(x.T,x)
    return Lambda * (np.linalg.norm(x, axis=-1) ** gamma) * (a - b)

def square_collision_kernel_A(Lambda,d,x,gamma):
    'x是(1,d)矩阵，输出(d,d)矩阵'
    a = np.linalg.norm(x, axis=-1) ** 2 * np.eye(d)
    b = np.matmul(x.T, x)
    return np.power(Lambda,0.5) * (np.linalg.norm(x, axis=-1) ** (-1+gamma/2)) * (a - b)

def random_group(data, group_size=2):
    # 打乱数据顺序
    random.shuffle(data)

    # 计算每组的元素数量
    num_groups = len(data) // group_size

    # 初始化分组结果
    groups = []

    # 按组分配数据
    for i in range(num_groups):
        start_index = i * group_size
        end_index = (i + 1) * group_size if i != num_groups - 1 else len(data)
        group = data[start_index:end_index]
        groups.append(group)
    return np.reshape(groups,(-1,2))

def batch_simulate_circle_BM(z_batch, t_batch):
    batch_size, d = z_batch.shape
    angles = np.arctan2(z_batch[:, 1], z_batch[:, 0]).reshape(batch_size, 1)
    #angles = torch.atan2(z_batch[:, 1], z_batch[:, 0]).view(batch_size, 1)

    # 添加高斯噪声
    noise_mean = np.zeros([batch_size, 1])  # 高斯噪声的均值
    noise_std = np.sqrt(t_batch)  # 高斯噪声的标准差
    noise = np.random.normal(noise_mean, noise_std)
    #noise_mean = torch.zeros([batch_size, 1])  # 高斯噪声的均值
    #noise_std = torch.sqrt(t_batch)  # 高斯噪声的标准差
    #noise = torch.normal(noise_mean, noise_std)
    noisy_angles = angles + noise

    # 将角度还原到两维的单位圆周上
    # 使用 sin 和 cos 函数来计算新的 (x, y) 坐标
    noisy_z_batch = np.concatenate((np.cos(noisy_angles), np.sin(noisy_angles)), axis=1)
    #noisy_z_batch = torch.stack((np.cos(noisy_angles), np.sin(noisy_angles)), dim=1)
    return noisy_z_batch.reshape(batch_size, d)

class two_dimension_Maxwell_SBM_solution():
    def __init__(self,T,L,n,Lambda,gamma,epsilon,sigma=0.7):
        #T是时间上界，L是速度分量截断值，n+1为[-L,L]取点数(包括两端)，h为速度步长，Lambda,gamma为Landau方程参数(Lambda需检验是否需要乘2)
        #epsilon为磨光核参数，sigma为磨光核计算时距离阈值(减少计算时间)，self.V为(n^2,2)速度格点中心矩阵
        self.epsilon=epsilon
        self.sigma=sigma
        self.d=2
        self.T=T
        self.L=L
        self.n=n
        self.Lambda=Lambda
        self.gamma=gamma
        self.a=np.linspace(-self.L,self.L,n+1,dtype=float)
        self.h=2*L/n
        self.a = np.linspace(-self.L+self.h/2, self.L-self.h/2, n , dtype=float)
        self.vx, self.vy= np.meshgrid(self.a, self.a)
        self.V = np.stack((self.vx, self.vy), axis=2).reshape(-1, self.d)

    def exact_solution(self,T):
        '输出T时刻、[-L,L]^2速度空间格点中心处的点(n^2,1)质量矩阵'
        K=1-np.exp(-T/8)/2
        z=np.linalg.norm(self.V,axis=-1)**2
        return (1/(2*np.pi*K)*np.exp(-z/(2*K))*(2-1/K+(1-K)/(2*K**2)*z)).reshape([-1,1])

    def compute_parameter(self,V,num,T):
        'num是粒子总数，T是时间，epsilon为磨光常数，计算格点质量分布时丢掉距离≥0.7的点，以便加速计算误差，但不会影响SBM精度'
        Z = np.zeros(shape=[self.n ** self.d, 1])
        for i in range(self.n ** self.d):
            x = np.repeat(self.V[i].reshape(1, -1), num, axis=0) - V
            z = np.linalg.norm(x, axis=-1)
            choice = np.where(z < self.sigma)
            x = x[choice]
            Z[i] = np.sum(mollifier(d=self.d, x=x, epsilon=epsilon), axis=0) / num

        mass = np.sum(Z)
        momentum = np.sum(np.multiply(Z, self.V), axis=0).reshape(1, -1)
        energy_grid = 0.5 * np.sum(np.multiply(Z, np.sum(np.power(self.V, 2), axis=-1).reshape(-1, 1)))
        energy_particle = 0.5 / num * np.sum(np.power(V, 2))
        z_T = self.exact_solution(T=T)
        relative_L2_error = np.linalg.norm(Z - z_T) / np.linalg.norm(z_T)

        indices=np.where(Z>0)
        entropy = np.sum(np.multiply(Z[indices].ravel(), np.log(Z[indices]).ravel()))

        indices=np.where(np.multiply(Z,np.power(z_T,-1)).ravel()>0)
        relative_entropy = np.sum(np.multiply(Z.ravel()[indices], (np.log(np.multiply(Z.ravel()[indices],np.power(z_T.ravel()[indices],-1))))))

        return [mass,momentum,energy_grid,energy_particle,relative_L2_error,entropy,relative_entropy]

    def solve(self,dt,V):
        'dt为时间步长，V为(length,2)每次循环中的速度分布矩阵，length是粒子总数，Z为(self.n**self.d,1)的格点取值矩阵，solve最终输出为(length,2)矩阵'
        length=V.shape[0]

        mass,energy_grid,energy_particle,relative_L2_error,entropy,relative_entropy=[],[],[],[],[],[]
        # 计算初始参量
        parameter=self.compute_parameter(V=V,num=length,T=0)
        mass.append(parameter[0])
        energy_grid.append(parameter[2])
        energy_particle.append(parameter[3])
        relative_L2_error.append(parameter[4])
        entropy.append(parameter[5])
        relative_entropy.append(parameter[-1])
        momentum=parameter[1]

        V_temporary=np.zeros_like(V)
        Time=0
        for t in range(int(self.T / dt)):
            time_start = time.time()
            group = random_group(list(range(length)), group_size=2)
            for i in range(int(length / 2)):
                p1 = group[i, 0]
                p2 = group[i, -1]
                V1 = V[p1].reshape(1, -1)
                V2 = V[p2].reshape(1, -1)
                z = V1 - V2
                z_unit = z / np.linalg.norm(z)
                q = np.array(batch_simulate_circle_BM(z_batch=np.array(z_unit),
                    t_batch=np.array((2 * self.Lambda ** 0.5 * np.linalg.norm(z) ** (self.gamma / 2)) ** 2 * dt)))
                z__ = (q * np.linalg.norm(z) - z) / 2
                V_temporary[p1] = V1 + z__.reshape([-1])
                V_temporary[p2] = V2 - z__.reshape([-1])
            V = V_temporary
            time_end = time.time()
            Time=Time+time_end-time_start

            if t%DT==(DT-1):
                # 计算每Dt时间的参量
                parameter = self.compute_parameter(V=V, num=length, T=(t+1)*dt)
                mass.append(parameter[0])
                energy_grid.append(parameter[2])
                energy_particle.append(parameter[3])
                relative_L2_error.append(parameter[4])
                entropy.append(parameter[5])
                relative_entropy.append(parameter[-1])
                momentum = np.concatenate([momentum,parameter[1]])

            print(f't={np.round((t+1)*dt,3)} finished')

            if t % int(1/dt) == int(1/dt)-1:
                np.save(f'T={(t+1)*dt}_dt={dt}_n={self.n}_N={length}_epsilon={self.epsilon}_2d-Maxwell_SBM_velocity', V)

        return V,self.h**self.d*np.array(mass),self.h**self.d*momentum.reshape(-1,self.d),self.h**self.d*np.array(energy_grid),energy_particle,\
            relative_L2_error,self.h**self.d*np.array(entropy),self.h**self.d*np.array(relative_entropy),Time


def plot(T,V,n,v,dt):
    'T为画图时刻，V为(length,1)速度矩阵，n为[-v,v]取点数,v为画图最后不要的点的分量上界'
    length = V.shape[0]
    Z = np.zeros(shape=[n**2, 1])
    a = np.linspace(-v, v, n, dtype=float)
    VX, VY = np.meshgrid(a, a)
    V_ = np.stack((VX, VY), axis=2).reshape([-1, 2])
    z_ = np.linalg.norm(V_, axis=-1) ** 2
    K = 1 - np.exp(-T / 8) / 2
    Z_=(1 / (2 * np.pi * K) * np.exp(-z_ / (2 * K)) * (2 - 1 / K + (1 - K) / (2 * K ** 2) * z_)).reshape([-1, n])

    p1=np.zeros(shape=[n,1])
    p2=a.reshape(-1,1)
    P=np.concatenate((p1,p2),axis=1)
    ZZ=np.zeros(shape=[n, 1])
    for i in range(n):
        x = np.repeat(P[i].reshape(1, -1), length, axis=0) - V
        z = np.linalg.norm(x, axis=-1)
        choice = np.where(z < 0.7)
        x = x[choice]
        ZZ[i] = np.sum(mollifier(d=2, x=x, epsilon=epsilon).reshape(-1, 1), axis=0) / length
    plt.plot(a,ZZ,label='SBM')
    plt.plot(a,(1 / (2 * np.pi * K) * np.exp(-p2** 2 / (2 * K)) * (2 - 1 / K + (1 - K) / (2 * K ** 2) * p2** 2)).ravel(),label='analytical')
    plt.title('cross-section')
    plt.xlabel('v_y')
    plt.ylabel('f')
    plt.legend()
    plt.savefig(f'T={T}_dt={dt}_n={n}_N={N}_epsilon={epsilon}_2d-Maxwell_SBM_cross-section.png',dpi=160)
    plt.close()
    # plt.show()

    for i in range(n**2):
        x = np.repeat(V_[i].reshape(1, -1), length, axis=0) - V
        z = np.linalg.norm(x, axis=-1)
        choice = np.where(z < 0.7)
        x = x[choice]
        Z[i] = np.sum(mollifier(d=2, x=x, epsilon=epsilon).reshape(-1, 1), axis=0) / length
    Z = Z.reshape([-1, n])
    fig = plt.figure(figsize=(17, 8))
    ax = fig.add_subplot(1,2,1,projection='3d')
    ax.plot_surface(VX, VY, Z-Z_, cmap=plt.cm.winter, alpha=1)
    ax.set_xlabel('v_x', fontsize=15)
    ax.set_ylabel('v_y', fontsize=15)
    ax.set_zlabel('n', fontsize=15)
    ax.set_title('SBM-analytic')

    ay = fig.add_subplot(1, 2, 2, projection='3d')
    ay.plot_surface(VX, VY, Z, cmap=plt.cm.winter,alpha=1)
    ay.set_xlabel('v_x', fontsize=15)
    ay.set_ylabel('v_y', fontsize=15)
    ay.set_zlabel('n', fontsize=15)
    ay.set_title('SBM')
    plt.savefig(f'T={T}_dt={dt}_n={n}_N={N}_epsilon={epsilon}_2d-Maxwell_SBM_3d-graph.png',dpi=160)
    plt.close()
    # plt.show()


T=1
v=4
n=100
dt=0.2
Dt=1
point=np.load('随机取样的点\\100000_points_2d-Maxwell.npy')[:10000]
N=point.shape[0]
DT=int(Dt/dt)
epsilon=0.01
Lambda=1/16
Lambda=Lambda*2
gamma=0
print(f'总时间：{T}')
print(f'速度空间分量最大值：{v}')
print(f'粒子数：{N}')
print(f'每个维度上的网格数(用于计算误差)：{n}')
print(f'时间步长：{dt}')
print(f'计算参量时间间隔：{Dt}')
print(f'磨光核参数：{epsilon}')
print(f'磨光核计算时距离阈值(减少计算时间)：0.7(默认)')


def save_and_draw(T,v,n,point,N,epsilon,dt):
    t = np.linspace(0, T, num=1 + int(T/Dt))
    Y = two_dimension_Maxwell_SBM_solution(T=T, L=v, n=n, Lambda=Lambda, gamma=gamma,epsilon=epsilon)
    V, mass, momentum, energy1, energy2, relative_L2_error, entropy, relative_entropy, Time = Y.solve(dt=dt,V=point)

    np.save(f'T={T}_dt={dt}_n={n}_N={N}_epsilon={epsilon}_2d-Maxwell_SBM_grid-mass', mass)
    np.save(f'T={T}_dt={dt}_n={n}_N={N}_epsilon={epsilon}_2d-Maxwell_SBM_grid-momentum', momentum)
    np.save(f'T={T}_dt={dt}_n={n}_N={N}_epsilon={epsilon}_2d-Maxwell_SBM_grid-energy', energy1)
    np.save(f'T={T}_dt={dt}_n={n}_N={N}_epsilon={epsilon}_2d-Maxwell_SBM_particle-energy', energy2)
    np.save(f'T={T}_dt={dt}_n={n}_N={N}_epsilon={epsilon}_2d-Maxwell_SBM_relative-L2-error',relative_L2_error)
    np.save(f'T={T}_dt={dt}_n={n}_N={N}_epsilon={epsilon}_2d-Maxwell_SBM_entropy', entropy)
    np.save(f'T={T}_dt={dt}_n={n}_N={N}_epsilon={epsilon}_2d-Maxwell_SBM_relative-entropy(SBM_vs_analytic)',relative_entropy)
    np.save(f'T={T}_dt={dt}_n={n}_N={N}_epsilon={epsilon}_2d-Maxwell_SBM_total-time', Time)

    #图片默认存成png格式
    plt.plot(t, mass)
    plt.xlabel("T")
    plt.ylabel('Mass')
    plt.title('Mass')
    plt.hlines(y=mass[0], xmin=0, xmax=T, colors='r', linestyles='--')
    plt.savefig(f'T={T}_dt={dt}_n={n}_N={N}_epsilon={epsilon}_2d-Maxwell_SBM_grid-mass.png', dpi=160)
    plt.close()
    # plt.show()
    plt.plot(t, momentum[:,0], label='momentum_x')
    plt.plot(t, momentum[:,1], label='momentum_y')
    plt.legend()
    plt.xlabel("T")
    plt.ylabel('momentum')
    plt.title('momentum')
    plt.savefig(f'T={T}_dt={dt}_n={n}_N={N}_epsilon={epsilon}_2d-Maxwell_SBM_grid-momentum.png', dpi=160)
    plt.close()
    # plt.show()
    plt.plot(t, energy1)
    plt.xlabel("T")
    plt.ylabel('Energy')
    plt.title('Energy')
    plt.hlines(y=energy1[0], xmin=0, xmax=T, colors='r', linestyles='--')
    plt.savefig(f'T={T}_dt={dt}_n={n}_N={N}_epsilon={epsilon}_2d-Maxwell_SBM_grid-energy.png', dpi=160)
    plt.close()
    # plt.show()
    plt.plot(t, energy2)
    plt.xlabel("T")
    plt.ylabel('Energy')
    plt.title('Energy')
    plt.hlines(y=energy2[0], xmin=0, xmax=T, colors='r', linestyles='--')
    plt.savefig(f'T={T}_dt={dt}_n={n}_N={N}_epsilon={epsilon}_2d-Maxwell_SBM_particle-energy.png', dpi=160)
    plt.close()
    # plt.show()
    plt.plot(t, relative_L2_error)
    plt.xlabel("T")
    plt.ylabel('relative L2 error')
    plt.title('relative L2 error')
    plt.savefig(f'T={T}_dt={dt}_n={n}_N={N}_epsilon={epsilon}_2d-Maxwell_SBM_relative-L2-error.png',dpi=160)
    plt.close()
    # plt.show()
    plt.plot(t, entropy)
    plt.xlabel("T")
    plt.ylabel('entropy')
    plt.title('entropy')
    plt.savefig(f'T={T}_dt={dt}_n={n}_N={N}_epsilon={epsilon}_2d-Maxwell_SBM_entropy.png', dpi=160)
    plt.close()
    # plt.show()
    plt.plot(t, relative_entropy)
    plt.xlabel("T")
    plt.ylabel('relative entropy')
    plt.title('relative entropy')
    plt.savefig(f'T={T}_dt={dt}_n={n}_N={N}_epsilon={epsilon}_2d-Maxwell_SBM_relative-entropy(SBM_vs_analytic).png',dpi=160)
    plt.close()
    # plt.show()

    plot(T=T, V=V, n=n, v=v,dt=dt)

    print( 'ok')

save_and_draw(T=T,v=v,n=n,point=point,N=N,epsilon=epsilon,dt=dt)