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

def shape_function(x,delta_x):
    z=np.zeros_like(x)
    X=np.abs(x)
    indice=np.where(X<=delta_x)
    z[indice]=1-X[indice]/delta_x
    return z


class two_dimension_plus_one_dimension_Landau_damping_SBM_solution():
    def __init__(self,T,L,n,n0,Lambda,gamma,epsilon,sigma=0.7):
        #T是时间上界，L是速度分量截断值，2L是空间区域，也等于总质量与总电量，n+1为[-L,L]取点数(包括两端)，h为速度步长，Lambda,gamma为Landau方程参数(Lambda需检验是否需要乘2)
        #epsilon为磨光核参数，sigma为磨光核计算时距离阈值(减少计算时间)，alpha为初始扰动参数，n0是画图的格点，self.V为(n^3,3)相空间格点中心矩阵
        self.epsilon = epsilon
        self.sigma = sigma
        self.d_of_D=1
        self.d_of_V=2
        self.D_all=self.d_of_D+self.d_of_V
        self.T=T
        self.L=L
        self.total=2*self.L
        self.n=n
        self.Lambda=Lambda
        self.gamma=gamma
        self.alpha=alpha
        #self.s=1e-6
        self.h=2*self.L/self.n

        grid = np.mgrid[-self.L + self.h / 2: self.L + self.h / 2: self.h,
            -self.L + self.h / 2: self.L + self.h / 2: self.h,
            self.h / 2: 2*self.L + self.h / 2: self.h]
        self.vx = grid[0].reshape([-1, 1])
        self.vy = grid[1].reshape([-1, 1])
        self.x = grid[2].reshape([-1, 1])
        self.all = np.concatenate((self.vx, self.vy, self.x), axis=1)
        self.V = self.all[:,0:2]
        self.X = self.all[:,2:]

        self.n0=n0
        self.h0 = 2 * self.L / self.n0
        grid_ = np.mgrid[-self.L + self.h0 / 2: self.L + self.h0 / 2: self.h0,
               self.h0 / 2: 2 * self.L + self.h0 / 2: self.h0]
        self.v1 = grid_[0].reshape([-1, 1])
        self.x1 = grid_[1].reshape([-1, 1])

        grid__ = np.mgrid[self.h0 / 2: self.L + self.h0 / 2: self.h0,
                self.h0 / 2: 2 * self.L + self.h0 / 2: self.h0]
        self.v2 = grid__[0].reshape([-1, 1])
        self.x2 = grid__[1].reshape([-1, 1])

    def initial_solution(self):
        '输出初始时刻T=0、[-L,L]^2X[0,2L]速度与距离空间中心上取的点(n^3,1)质量矩阵'
        z = np.linalg.norm(self.V, axis=-1) ** 2
        return 1/(2*np.pi)*np.exp(-z/2)*(1+self.alpha*np.sin(self.X))/(2*self.L)

    def compute_parameter(self,all,num):
        'num是粒子总数，epsilon为磨光常数，计算格点质量分布时丢掉距离≥0.7的点，以便加速计算误差，但不会影响SBM精度'
        Z = np.zeros(shape=[self.n ** self.D_all, 1])
        for i in range(self.n ** self.D_all):
            x = np.repeat(self.all[i].reshape(1, -1), num, axis=0) - all
            z = np.linalg.norm(x, axis=-1)
            choice = np.where(z < self.sigma)
            x = x[choice]
            Z[i] = np.sum(mollifier(d=self.D_all, x=x, epsilon=epsilon), axis=0) / num

        mass = np.sum(Z)
        momentum = np.sum(np.multiply(Z, self.V), axis=0).reshape(1, -1)
        energy_grid = 0.5 * np.sum(np.multiply(Z, np.sum(np.power(self.V_, 2), axis=-1).reshape(-1, 1)))

        indices=np.where(Z>0)
        entropy = np.sum(np.multiply(Z[indices].ravel(), np.log(Z[indices]).ravel()))

        return [mass,momentum,energy_grid,entropy]

    def solve(self,dt,all,m=5,measure=False):
        #dt为时间步长，all为(length,3)每次循环中的速度空间分布矩阵，length是粒子总数，Z为(self.n**self.d,1)的格点取值矩阵
        #V为每次循环中的速度分布矩阵，X为每次循环中的空间分布矩阵，solve最终输出为(length,3)矩阵，m是迭代次数，measure=True则测量每次的质量、动量与熵
        V=all[:,0:2]
        X=all[:,2:]
        length=all.shape[0]

        mass, energy_grid, entropy = [], [], []
        momentum=np.zeros(shape=[1,self.d_of_V])
        if measure==True:
            # 计算初始参量
            parameter = self.compute_parameter(all=all, num=length)
            mass.append(parameter[0])
            energy_grid.append(parameter[2])
            entropy.append(parameter[-1])
            momentum = parameter[1]
        particle_energy=[0.5*self.total / length * np.linalg.norm(V) ** 2]
        Time = 0
        time_start = time.time()

        z=np.zeros(shape=[length,self.n])
        for i in range(self.n):
            z[:,i]=X[:,0]-self.X[i]
        indice=np.where(z>=2*self.L-self.h)
        z[indice]=z[indice]-2*self.L
        indice = np.where(z <= -2 * self.L + self.h)
        z[indice] = z[indice] + 2 * self.L
        z=shape_function(z,self.h)

        #F是格子中心处电荷密度，fai是格子中心处电势
        F = (np.sum(z,axis=0)/ (length * self.h) - 0.5 / self.L)*self.total
        F_hat = np.fft.fft(F)
        fai_hat = np.zeros_like(F_hat)
        for i in range(1, int((self.n + 1) / 2)):
            fai_hat[i] = F_hat[i] / ((0.5*i) ** 2)
            fai_hat[-i] = F_hat[-i] / ((0.5*i) ** 2)
        fai = np.fft.ifft(fai_hat).real
        E = -(np.roll(fai,-1) - np.roll(fai,1)) / (2 * self.h)
        time_end = time.time()
        Time = Time + time_end - time_start
        electric_energy=[np.linalg.norm(E)**2*self.h/2]

        for t in range(int(self.T / dt)):
            time_start = time.time()
            #粒子碰撞
            X_belong_position=np.digitize(X.ravel(), np.linspace(0, 2 * self.L, self.n + 1))-1
            for j in range(self.n):
                indices = np.where(X_belong_position == j)
                V_tem=V[indices]
                #before_length是每个格子中的粒子数
                before_length=V_tem.shape[0]
                extra=np.zeros(shape=2)
                #随机选取一个粒子
                choice = np.random.randint(before_length)
                if before_length % 2!=0:
                    extra = V_tem[choice]
                    V_tem=np.concatenate((V_tem[:choice],V_tem[choice+1:]),axis=0)
                group = random_group(list(range(V_tem.shape[0])), group_size=2)
                for i in range(int(V_tem.shape[0] / 2)):
                    p1 = group[i, 0]
                    p2 = group[i, -1]
                    V1 = V_tem[p1].reshape(1, -1)
                    V2 = V_tem[p2].reshape(1, -1)
                    z = V1 - V2
                    z_unit = z / np.linalg.norm(z)
                    q = np.array(batch_simulate_circle_BM(z_batch=np.array(z_unit),
                        t_batch=np.array((2 * self.Lambda ** 0.5 * np.linalg.norm(z) ** (self.gamma / 2)) ** 2 * dt)))
                    z__ = (q * np.linalg.norm(z) - z) / 2
                    V_tem[p1] = V1 + z__.ravel()
                    V_tem[p2] = V2 - z__.ravel()
                if before_length % 2!=0:
                    V_tem=np.concatenate((V_tem[:choice],extra.reshape([1,-1]),V_tem[choice:]),axis=0)
                V[indices]=V_tem

            #粒子输运
            X_temporary=np.zeros_like(X)
            V_temporary=np.zeros_like(V)
            E_temporary=np.zeros_like(E)

            X_belong_position = np.digitize(X.ravel(), np.linspace(0, 2 * self.L, self.n + 1))-1
            E_each_particle = E[X_belong_position]
            #给出粒子输运猜测值(实际上用的是shape function取特征函数)
            X_temporary[:, 0] = X[:, 0] + V[:, 0] * dt + 0.5 * E_each_particle * dt ** 2
            V_temporary[:, 0] = V[:, 0] + E_each_particle * dt

            for i in range(m):
                X_half=(0.5*(X+X_temporary))%(2*self.L)
                V_half=0.5*(V+V_temporary)
                #X_belong_position_half = np.digitize(X_half.ravel(), np.linspace(0, 2 * self.L, self.n + 1)) - 1
                z=np.zeros(shape=[length,self.n])
                for i in range(self.n):
                    z[:, i] = X_half[:, 0] - self.X[i]
                indice = np.where(z >= 2 * self.L - self.h)
                z[indice] = z[indice] - 2 * self.L
                indice = np.where(z <= -2 * self.L + self.h)
                z[indice] = z[indice] + 2 * self.L
                z = shape_function(z, self.h)
                J = np.zeros(shape=[self.n])
                for i in range(self.n):
                    #indices = np.where(X_belong_position_half == i)
                    indice=np.where(z[:,i]!=0)
                    J[i]=np.sum(np.multiply(z[indice,i],V_half[indice,0]))
                    #if indices[0].shape[0] > 0:
                    #    J[i] = np.sum(V_half[indices, 0])
                J = J / (self.h * length) * self.total
                J_mean = np.sum(J) / self.n
                E_temporary = E + dt * (J_mean - J)
                E_half=0.5*(E+E_temporary)

                #E_half_each_particle = E_half[X_belong_position_half]
                E_half_each_particle = np.sum(np.multiply(z,E_half.reshape(1,-1)),axis=1)
                X_temporary[:, 0] = X[:, 0] + 0.5*(V[:,0]+V_temporary[:,0]) * dt
                V_temporary[:, 0] = V[:, 0] + E_half_each_particle * dt

            X[:,0]=np.copy(X_temporary[:,0])%(2*self.L)
            V[:,0]=np.copy(V_temporary[:,0])
            E=np.copy(E_temporary)

            all=np.concatenate((V,X),axis=1)

            time_end = time.time()
            Time = Time + time_end - time_start


            if t%DT==(DT-1):
                if measure == True:
                    parameter = self.compute_parameter(all=all, num=length)
                    mass.append(parameter[0])
                    energy_grid.append(parameter[2])
                    entropy.append(parameter[-1])
                    momentum = np.concatenate([momentum, parameter[1]])
                particle_energy.append(0.5 * self.total / length * np.linalg.norm(V) ** 2)


                #Z = Z.reshape(-1, self.n)
                #F = self.h ** self.d_of_V * np.sum(Z, axis=0) - 0.5 / self.L

                electric_energy.append(np.linalg.norm(E) ** 2*self.h/2)

            print(f't={np.round((t + 1) * dt, 3)} finished')

            if t % int(1 / dt) == int(1 / dt) - 1:
                np.save(f'T={(t + 1) * dt}_dt={dt}_n={self.n}_N={length}_epsilon={self.epsilon}_alpha={self.alpha}_Lambda={self.Lambda}_Landau_damping_1D_2V_SBM_phase', all)

                Z = np.zeros(shape=[self.n0 ** self.d_of_V, 1])
                all_x = np.concatenate((self.v1, Z, self.x1), axis=1)
                all_y = np.concatenate((Z, self.v1, self.x1), axis=1)
                for i in range(self.n0 ** self.d_of_V):
                    x = np.repeat(all_x[i].reshape(1, -1), length, axis=0) - all
                    x=np.concatenate((x[:,0].reshape(-1,1),x[:,-1].reshape(-1,1)),axis=1)
                    z = np.linalg.norm(x, axis=-1)
                    choice = np.where(z < 0.7)
                    x = x[choice]
                    Z[i] = np.sum(mollifier(d=self.d_of_V, x=x, epsilon=epsilon), axis=0)
                Z = Z / length
                self.drawing1(Z.reshape(-1, self.n0),(t + 1) * dt,length)

                for i in range(self.n0 ** self.d_of_V):
                    x = np.repeat(all_y[i].reshape(1, -1), length, axis=0) - all
                    x = np.copy(x[:,1:]).reshape(-1,2)
                    z = np.linalg.norm(x, axis=-1)
                    choice = np.where(z < 0.7)
                    x = x[choice]
                    Z[i] = np.sum(mollifier(d=self.d_of_V, x=x, epsilon=epsilon), axis=0)
                Z = Z / length
                self.drawing2(Z.reshape(-1, self.n0),(t + 1) * dt,length)

                Z = np.zeros(shape=[int(0.5 * self.n0 ** self.d_of_V), 1])
                norm_all = np.concatenate((np.linalg.norm(V, axis=-1).reshape(-1, 1), X), axis=1)
                all_norm = np.concatenate((self.v2, self.x2), axis=1)
                for i in range(int(0.5 * self.n0 ** 2)):
                    x = np.repeat(all_norm[i].reshape(1, -1), length, axis=0) - norm_all
                    z = np.linalg.norm(x, axis=-1)
                    choice = np.where(z < 0.7)
                    x = x[choice]
                    Z[i] = np.sum(mollifier(d=self.d_of_V, x=x, epsilon=epsilon), axis=0)
                Z = Z / length
                self.drawing3(Z.reshape(-1, self.n0),(t + 1) * dt,length)

        if measure == True:
            return all,self.h**self.D_all*np.array(mass),self.h**self.D_all*momentum.reshape(-1,self.d_of_V),self.h**self.D_all*np.array(energy_grid),\
                particle_energy,electric_energy, self.h**self.D_all*np.array(entropy),Time
        else:
            return all,particle_energy, electric_energy,Time


    def drawing1(self,matrix,T,N):
        np.save(f'T={T}_dt={dt}_n={self.n}_N={N}_n0={n0}_epsilon={self.epsilon}_alpha={self.alpha}_Lambda={self.Lambda}_Landau_damping_1D_2V_v_x_distribution_in_space',matrix)
        plt.imshow(matrix, cmap='jet', origin='lower', extent=[0, 2*self.L, -self.L, self.L])
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('v_x')
        plt.title('v_x distribution in space')
        #plt.show()
        plt.savefig(f'T={T}_dt={dt}_n={self.n}_N={N}_n0={n0}_epsilon={self.epsilon}_alpha={self.alpha}_Lambda={self.Lambda}_Landau_damping_1D_2V_v_x_distribution_in_space_graph.eps',bbox_inches='tight',dpi=160)
        plt.close()

    def drawing2(self,matrix,T,N):
        np.save(f'T={T}_dt={dt}_n={self.n}_N={N}_n0={n0}_epsilon={self.epsilon}_alpha={self.alpha}_Lambda={self.Lambda}_Landau_damping_1D_2V_v_y_distribution_in_space',matrix)
        plt.imshow(matrix, cmap='jet', origin='lower', extent=[0, 2*self.L, -self.L, self.L])
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('v_y')
        plt.title('v_y distribution in space')
        #plt.show()
        plt.savefig(f'T={T}_dt={dt}_n={self.n}_N={N}_n0={n0}_epsilon={self.epsilon}_alpha={self.alpha}_Lambda={self.Lambda}_Landau_damping_1D_2V_v_y_distribution_in_space_graph.eps',bbox_inches='tight',dpi=160)
        plt.close()

    def drawing3(self,matrix,T,N):
        np.save(f'T={T}_dt={dt}_n={self.n}_N={N}_n0={n0}_epsilon={self.epsilon}_alpha={self.alpha}_Lambda={self.Lambda}_Landau_damping_1D_2V_v_module_distribution_in_space',matrix)
        plt.imshow(matrix, cmap='jet', origin='lower', extent=[0, 2*self.L, 0, 2**0.5*self.L])
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('|v|')
        plt.title('|v| distribution in space')
        #plt.show()
        plt.savefig(f'T={T}_dt={dt}_n={self.n}_N={N}_n0={n0}_epsilon={self.epsilon}_alpha={self.alpha}_Lambda={self.Lambda}_Landau_damping_1D_2V_v_module_distribution_in_space_graph.eps',bbox_inches='tight',dpi=160)
        plt.close()



T=10
v=2*np.pi
n=128
n0=200
dt=0.02
Dt=0.02
alpha=0.1
point=np.load(f'随机取样的点\\500000_points_Landau_damping_1D_2V_alpha={alpha}.npy')[:500000]
N=point.shape[0]
DT=int(Dt/dt)
epsilon=0.01
Lambda=0.1
gamma=-2
#初始分布标准差
Standard_Deviation=0.5
print(f'总时间：{T}')
print(f'速度空间分量最大值：{v}')
print(f'空间范围：[0, {2*v}]')
print(f'粒子数：{N}')
print(f'每个维度上的网格数(用于计算误差)：{n}')
print(f'每个维度上的网格数(用于画粒子速度关于空间的分布)：{n0}')
print(f'时间步长：{dt}')
print(f'计算参量时间间隔：{Dt}')
print(f'磨光核参数：{epsilon}')
print(f'磨光核计算时距离阈值(减少计算时间)：0.7(默认)')
print(f'初始分布扰动程度：{alpha}')
print(f'碰撞强度：{Lambda}')
gamma_decay=-1/(0.5)**3*np.power(np.pi/8,0.5)*np.exp(-1.5-0.5*0.5**-2)-Lambda*(2/(9*np.pi))**0.5
print(f'电场指数衰减系数：{gamma_decay}')


#保存到npz文件，且算相对误差
def save_and_draw(T,v,n,point,N,epsilon,dt):
    t = np.linspace(0, T, num=1 + int(T/Dt))
    Y = two_dimension_plus_one_dimension_Landau_damping_SBM_solution(T=T, L=v, n=n, n0=n0, Lambda=Lambda, gamma=gamma,epsilon=epsilon)
    all ,particle_energy ,electric_energy, Time= Y.solve(dt=dt,all=point)
    np.save(f'T={T}_dt={dt}_n={n}_N={N}_epsilon={epsilon}_alpha={alpha}_Lambda={Lambda}_Landau_damping_1D_2V_SBM_particle-energy', particle_energy)
    np.save(f'T={T}_dt={dt}_n={n}_N={N}_epsilon={epsilon}_alpha={alpha}_Lambda={Lambda}_Landau_damping_1D_2V_SBM_electric-energy', electric_energy)
    np.save(f'T={T}_dt={dt}_n={n}_N={N}_epsilon={epsilon}_alpha={alpha}_Lambda={Lambda}_Landau_damping_1D_2V_SBM_total-time', Time)

    plt.plot(t, particle_energy)
    plt.xlabel("T")
    plt.title('Kinetic energy')
    #plt.show()
    plt.savefig(f'T={T}_dt={dt}_n={n}_N={N}_epsilon={epsilon}_alpha={alpha}_Lambda={Lambda}_Landau_damping_1D_2V_SBM_kinetic-energy.png', dpi=160)
    plt.close()

    plt.plot(t, np.power(2*np.array(electric_energy),0.5),label='SBM')
    plt.xlabel("T")
    plt.title('Electric field L_2 norm')
    end=np.minimum(T,np.log(0.1)/gamma_decay)
    plt.plot(t[:int(end/Dt)+1],np.power(2*electric_energy[0],0.5)*np.exp(gamma_decay*t[:int(end/Dt)+1]),'--',label='the decay predicted by theory')
    plt.yscale('log')
    plt.legend()
    # plt.show()
    plt.savefig(f'T={T}_dt={dt}_n={n}_N={N}_epsilon={epsilon}_alpha={alpha}_Lambda={Lambda}_Landau_damping_1D_2V_SBM_Electric-field-L_2-norm.png', dpi=160)
    plt.close()

    plt.plot(t, np.array(electric_energy)+np.array(particle_energy))
    plt.xlabel("T")
    plt.title('Total energy')
    plt.hlines(y=(np.array(electric_energy)+np.array(particle_energy))[0], xmin=0, xmax=T, colors='r', linestyles='--')
    # plt.show()
    plt.savefig(f'T={T}_dt={dt}_n={n}_N={N}_epsilon={epsilon}_alpha={alpha}_Lambda={Lambda}_Landau_damping_1D_2V_SBM_total-energy.png', dpi=160)
    plt.close()

    print( 'ok')

save_and_draw(T=T,v=v,n=n,point=point,N=N,epsilon=epsilon,dt=dt)