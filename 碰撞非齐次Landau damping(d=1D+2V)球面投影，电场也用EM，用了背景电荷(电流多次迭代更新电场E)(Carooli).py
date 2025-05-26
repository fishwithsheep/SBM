import matplotlib.pyplot as plt
import numpy as np
import random
np.random.seed(10)
random.seed(10)


def psi_yipuxilong(d,x,yipuxilong):
    'x是(n,d)矩阵，输出(n,1)矩阵'
    return (2*np.pi*yipuxilong)**(-d/2)*np.exp(-np.linalg.norm(x,axis=-1)**2/(2*yipuxilong)).reshape(-1,1)

def gradient_psi_yipuxilong(d,x,yipuxilong):
    'x是(n,d)矩阵，输出(n,d)矩阵'
    return (2*np.pi*yipuxilong)**(-d/2)*np.multiply(np.exp(-np.linalg.norm(x,axis=-1)**2/(2*yipuxilong)).reshape(-1,1),(-x/yipuxilong))

def collision_kernel_A(Lambda,d,x,gamma):
    'x是(1,d)矩阵，输出(d,d)矩阵(太多了内存不行，而且随机用不到)'
    a=np.linalg.norm(x,axis=-1)**2*np.eye(d)
    b = np.matmul(x.T,x)
    return Lambda * (np.linalg.norm(x, axis=-1) ** gamma) * (a - b)

def square_collision_kernel_A(Lambda,d,x,gamma):
    'x是(1,d)矩阵，输出(d,d)矩阵(太多了内存不行，而且随机用不到)'
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

def shape_function(x,delta_x):
    z=np.zeros_like(x)
    X=np.abs(x)
    indice=np.where(X<=delta_x)
    z[indice]=1-X[indice]/delta_x
    return z

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

class two_dimension():
    def __init__(self,T,L,n,Lambda,gamma):
        'T是时间上界,v∈[-L,L]^2,n+1为[-L,L]取点数(包括两端),h为速度与空间步长,x∈[0,2*L],n+1为[0,2L]取点数(包括两端)'
        self.d_of_D=1
        self.d_of_V=2
        self.D_all=self.d_of_D+self.d_of_V
        self.T=T
        self.L=L
        self.n=n
        self.Lambda=Lambda
        self.gamma=gamma
        self.alpha=alpha
        self.s=1e-6
        self.h=2*self.L/self.n
        self.grid_V=np.linspace(-self.L,self.L,self.n+1)
        self.grid_D=np.linspace(0,2*self.L,self.n+1)

        grid = np.mgrid[-self.L + self.h / 2: self.L + self.h / 2: self.h,
            -self.L + self.h / 2: self.L + self.h / 2: self.h,
            self.h / 2: 2*self.L + self.h / 2: self.h]
        self.vx = grid[0].reshape([-1, 1])
        self.vy = grid[1].reshape([-1, 1])
        self.x = grid[2].reshape([-1, 1])
        self.all = np.concatenate((self.vx, self.vy, self.x), axis=1)
        self.V = self.all[:,0:2]
        self.X = self.all[:,2:]
        print(self.all.shape)
        print(self.V.shape)
        print(self.X.shape)

        self.n0=n0
        self.h0 = 2 * self.L / self.n0
        grid0 = np.mgrid[-self.L + self.h0 / 2: self.L + self.h0 / 2: self.h0,
               self.h0 / 2: 2 * self.L + self.h0 / 2: self.h0]
        self.v0 = grid0[0].reshape([-1, 1])
        self.x0 = grid0[1].reshape([-1, 1])

        grid1 = np.mgrid[self.h0 / 2: self.L + self.h0 / 2: self.h0,
                self.h0 / 2: 2 * self.L + self.h0 / 2: self.h0]
        self.v1 = grid1[0].reshape([-1, 1])
        self.x1 = grid1[1].reshape([-1, 1])

    def exact_solution(self):
        '输出初始时刻T=0、[-L,L]^2X[0,2L]速度与距离空间中心上取的点(n^3,1)质量矩阵'
        z = np.linalg.norm(self.V, axis=-1) ** 2
        return 1/(2*np.pi)*np.exp(-z/2)*(1+self.alpha*np.sin(self.X))/(2*self.L)

    def solve(self,dt,yipuxilong,input):
        'dt为时间步长，yipuxilong为常数(见论文)，input为(len,3)矩阵'
        'V为每次循环中的速度分布矩阵，输出为(len,2)矩阵'
        'X为每次循环中的空间分布矩阵，输出为(len,1)矩阵'
        '计算格点质量分布时丢掉距离≥0.7的点'
        '计算是否碰撞时看距离是否小于相对速度乘dt'
        'all为每次循环中的速度空间分布矩阵，输出为(len,3)矩阵'
        V=input[:,0:2]
        X=input[:,2:]
        all=input
        length=all.shape[0]
        num=self.n ** self.D_all

        #磨光核去算离散能量
        #Z = np.zeros(shape=[num, 1])
        #for i in range(num):
        #    x = np.repeat(self.all[i].reshape(1, -1), length, axis=0) - all
        #    z = np.linalg.norm(x, axis=-1)
        #    choice = np.where(z < 0.7)
        #    x = x[choice]
        #    Z[i] = np.sum(psi_yipuxilong(d=self.D_all ,x=x, yipuxilong=yipuxilong),axis=0)
       #     if i %10000==0 and i!=0:
       #         print(i)
        #Z=Z/length
        #mass=[(np.sum(Z))]
        mass=[0]
        #momentum = np.sum(np.multiply(Z, self.V), axis=0).reshape(1,-1)
        momentum=np.array([[0,1]])

        #energy1=[0.5 * np.sum(np.multiply(Z, (np.linalg.norm(self.V, axis=-1) ** 2).reshape(-1, 1)))]
        energy1=[0]

        energy2=[0.5*Q / length * np.linalg.norm(V) ** 2]

        #E_ = self.h ** self.D_all * Z
        #Indices = np.where(E_ != 0)
        #entropy1=[np.sum(np.multiply(E_[Indices].ravel(),np.log(E_[Indices]).ravel()))]
        entropy1=[0]

        H, edges = np.histogramdd(all,bins=[self.grid_V,self.grid_V,self.grid_D])
        H=H.ravel()/length
        indices=np.where(H!=0)
        H=H[indices]
        entropy2=[np.sum(np.multiply(H.ravel(),np.log(H*self.h**2).ravel()))]

        #E = np.multiply(Z, np.power(z_T, -1))
        #Indices = np.where(E != 0)
        #relative_entropy = [np.sum(np.multiply(Z[Indices].reshape([-1]), np.log(E[Indices]).reshape([-1])))]

        #Z = Z.reshape(-1,self.n)
        # F = self.h**self.d_of_V * np.sum(Z, axis=0) - 0.5 / self.L
        #F = (np.histogram(X[:, 0], np.linspace(0, 2 * self.L, self.n + 1))[0] / (length * self.h) - 0.5 / self.L)*Q

        z=np.zeros(shape=[length,self.n])
        for i in range(self.n):
            z[:,i]=X[:,0]-self.X[i]
        indice=np.where(z>=2*self.L-self.h)
        z[indice]=z[indice]-2*self.L
        indice = np.where(z <= -2 * self.L + self.h)
        z[indice] = z[indice] + 2 * self.L
        z=shape_function(z,self.h)
        F = (np.sum(z,axis=0)/ (length * self.h) - 0.5 / self.L)*Q

        F_hat = np.fft.fft(F)
        fai_hat = np.zeros_like(F_hat)
        for i in range(1, int((self.n + 1) / 2)):
            fai_hat[i] = F_hat[i] / ((0.5*i) ** 2)
            fai_hat[-i] = F_hat[-i] / ((0.5*i) ** 2)
        fai = np.fft.ifft(fai_hat).real

        E = -(np.roll(fai,-1) - np.roll(fai,1)) / (2 * self.h)
        electric_energy=[np.linalg.norm(E)**2*self.h/2]
        print(np.linalg.norm(E) ** 2*self.h/2)


        for t in range(int(self.T / dt)):
            X_belong_position=np.digitize(X.ravel(), np.linspace(0, 2 * self.L, self.n + 1))-1
            for j in range(self.n):
                indices = np.where(X_belong_position == j)
                V_tem=V[indices]
                before_length=V_tem.shape[0]
                extra=np.zeros(shape=2)
                if before_length % 2!=0:
                    index = np.random.randint(before_length)
                    extra = V_tem[index]    
                    V_tem=np.concatenate((V_tem[:index],V_tem[index+1:]),axis=0)
                    #extra=V_tem[-1]
                    #V_tem=V_tem[:-1]
                group = random_group(list(range(V_tem.shape[0])), group_size=2)
                for i in range(int(V_tem.shape[0] / 2)):
                    p1 = group[i, 0]
                    p2 = group[i, -1]
                    V1 = V_tem[p1].reshape(1, -1)
                    V2 = V_tem[p2].reshape(1, -1)
                    z = V1 - V2
                    z_unit = z / np.linalg.norm(z)
                    q = np.array(batch_simulate_circle_BM(z_batch=np.array(z_unit),
                                                          t_batch=np.array((2 * self.Lambda ** 0.5 * np.linalg.norm(
                                                              z) ** (self.gamma / 2)) ** 2 * dt)))
                    z__ = (q * np.linalg.norm(z) - z) / 2
                    V_tem[p1] = V1 + z__.reshape([-1])
                    V_tem[p2] = V2 - z__.reshape([-1])
                if before_length % 2!=0:
                    #V_tem=np.concatenate((V_tem,extra.reshape([1,-1])),axis=0)
                    V_tem=np.concatenate((V_tem[:index],extra.reshape([1,-1]),V_tem[index:]),axis=0)
                V[indices]=V_tem
            all=np.concatenate((V,X),axis=1)
            #Z=np.zeros(shape=[num, 1])
            #for i in range(num):
            #    x = np.repeat(self.all[i].reshape(1, -1), length, axis=0) - all
            #    z = np.linalg.norm(x, axis=-1)
            #    choice = np.where(z < 0.7)
            #    x = x[choice]
            #    Z[i] = np.sum(psi_yipuxilong(d=self.D_all, x=x, yipuxilong=yipuxilong), axis=0)
            #    if i % 10000 == 0 and i != 0:
            #        print(i)
            #Z=Z.reshape(-1,self.n)/ length
            #F = self.h**self.d_of_V * np.sum(Z, axis=0) - 0.5 / self.L

            X_temporary=np.zeros_like(X)
            V_temporary=np.zeros_like(V)
            E_temporary=np.zeros_like(E)

            X_belong_position = np.digitize(X.ravel(), np.linspace(0, 2 * self.L, self.n + 1))-1
            E_each_particle = E[X_belong_position]
            X_temporary[:, 0] = X[:, 0] + V[:, 0] * dt + 0.5 * E_each_particle * dt ** 2
            V_temporary[:, 0] = V[:, 0] + E_each_particle * dt


            for i in range(5):
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
                J = J / (self.h * length) * Q
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

            if t%DT==(DT-1):
                # 磨光核去算离散能量
                #Z = np.zeros(shape=[num, 1])
                #for i in range(num):
                #    x = np.repeat(self.all[i].reshape(1, -1), length, axis=0) - all
                #    z = np.linalg.norm(x, axis=-1)
                #    choice = np.where(z < 0.7)
                #    x = x[choice]
                #    Z[i] = np.sum(psi_yipuxilong(d=self.D_all, x=x, yipuxilong=yipuxilong), axis=0)
                #    if i % 10000 == 0 and i != 0:
                #        print(i)
                #Z=Z/ length
                #mass.append(np.sum(Z))
                #momentum=np.concatenate([momentum,np.sum(np.multiply(Z, self.V), axis=0).reshape(1,-1)])
                #energy1.append((0.5 * np.sum(np.multiply(Z, (np.linalg.norm(self.V, axis=-1) ** 2).reshape(-1, 1)))))

                energy2.append(0.5*Q / length * np.linalg.norm(V) ** 2)

                #E_ = self.h ** self.D_all * Z
                #Indices = np.where(E_ != 0)
                #entropy1.append(np.sum(np.multiply(E_[Indices].ravel(), np.log(E_[Indices]).ravel())))

                H, edges = np.histogramdd(all, bins=[self.grid_V, self.grid_V, self.grid_D])
                H = H.ravel() / length
                indices = np.where(H != 0)
                H = H[indices]
                entropy2.append(np.sum(np.multiply(H.ravel(), np.log(H*self.h**2).ravel())))

                #E = np.multiply(Z, np.power(z_T, -1))
                #Indices = np.where(E != 0)
                #relative_entropy.append(np.sum(np.multiply(Z[Indices].reshape([-1]), np.log(E[Indices]).reshape([-1]))))


                #Z = Z.reshape(-1, self.n)
                #F = self.h ** self.d_of_V * np.sum(Z, axis=0) - 0.5 / self.L

                electric_energy.append(np.linalg.norm(E) ** 2*self.h/2)
                print(np.linalg.norm(E) ** 2*self.h/2)

            print(t)


            if t==0:
                np.save(f'已经开始跑了', all)

            if t % int(50/dt) == int(50/dt)-1:
                np.save(f'T={(t+1)*dt},L={int(v/np.pi)}pi,dt={dt},n={n},N={N},n0={n0},Lambda={Lambda},α={alpha},yipuxilong={yipuxilong}非齐次Landau damping,d=1D+2V=3 blob', all)

        Z = np.zeros(shape=[self.n0**2, 1])
        all_x=np.concatenate((Z,self.v0,self.x0),axis=1)
        all_y = np.concatenate((self.v0,Z, self.x0), axis=1)
        for i in range(self.n0**2):
            x = np.repeat(all_x[i].reshape(1, -1), length, axis=0) - all
            z = np.linalg.norm(x, axis=-1)
            choice = np.where(z < 0.7)
            x = x[choice]
            Z[i] = np.sum(psi_yipuxilong(d=self.D_all, x=x, yipuxilong=yipuxilong), axis=0)
        Z = Z / length
        self.drawing1(Z.reshape(-1,self.n0))
        for i in range(self.n0**2):
            x = np.repeat(all_y[i].reshape(1, -1), length, axis=0) - all
            z = np.linalg.norm(x, axis=-1)
            choice = np.where(z < 0.7)
            x = x[choice]
            Z[i] = np.sum(psi_yipuxilong(d=self.D_all, x=x, yipuxilong=yipuxilong), axis=0)
        Z = Z / length
        self.drawing2(Z.reshape(-1, self.n0))
        Z = np.zeros(shape=[int(0.5*self.n0 ** 2), 1])
        norm_all=np.concatenate((np.linalg.norm(V,axis=-1).reshape(-1,1),X),axis=1)
        all_norm=np.concatenate((self.v1,self.x1),axis=1)
        for i in range(int(0.5*self.n0**2)):
            x = np.repeat(all_norm[i].reshape(1, -1), length, axis=0) - norm_all
            z = np.linalg.norm(x, axis=-1)
            choice = np.where(z < 0.7)
            x = x[choice]
            Z[i] = np.sum(psi_yipuxilong(d=2, x=x, yipuxilong=yipuxilong), axis=0)
        Z = Z / length
        self.drawing3(Z.reshape(-1, self.n0))


        return (V,X,self.h**self.D_all*np.array(mass),self.h**self.D_all*momentum.reshape(-1,self.d_of_V),self.h**self.D_all*np.array(energy1),
            energy2,electric_energy, np.array(entropy1),entropy2)


    def drawing1(self,matrix, title=None):
        plt.imshow(matrix, cmap='jet', origin='lower', extent=[0, 2*self.L, -self.L, self.L])
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('vx')
        plt.title(title)
        #plt.show()
        plt.savefig(
            f'T={T},L={int(v / np.pi)}pi,dt={dt},n={n},N={N},n0={n0},Lambda={Lambda},α={alpha},yipuxilong={yipuxilong}非齐次Landau damping,d=1D+2V=3 blob 速度vx与x figure.eps',
            bbox_inches='tight')
        plt.close()

    def drawing2(self,matrix, title=None):
        plt.imshow(matrix, cmap='jet', origin='lower', extent=[0, 2*self.L, -self.L, self.L])
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('vy')
        plt.title(title)
        #plt.show()
        plt.savefig(
            f'T={T},L={int(v / np.pi)}pi,dt={dt},n={n},N={N},n0={n0},Lambda={Lambda},α={alpha},yipuxilong={yipuxilong}非齐次Landau damping,d=1D+2V=3 blob 速度vy与x figure.eps',
            bbox_inches='tight')
        plt.close()

    def drawing3(self,matrix, title=None):
        plt.imshow(matrix, cmap='jet', origin='lower', extent=[0, 2*self.L, 0, 2**0.5*self.L])
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('v')
        plt.title(title)
        #plt.show()
        plt.savefig(
            f'T={T},L={int(v / np.pi)}pi,dt={dt},n={n},N={N},n0={n0},Lambda={Lambda},α={alpha},yipuxilong={yipuxilong}非齐次Landau damping,d=1D+2V=3 blob 速度v与x figure.eps',
            bbox_inches='tight')
        plt.close()



T=50
v=2*np.pi
n=128
#最后画热力图每个维数取点个数
n0=200
dt=0.02
Dt=0.02
alpha=0.1
print(int(T/dt))
point=np.load(f'/home/yuyang/Python/本科毕设/500000个点(非齐次Landau damping,d=1+2=3),α={alpha},Carolli文章.npy')
N=point.shape[0]
DT=int(Dt/dt)
print(N)
print(dt)
yipuxilong=0.01
#碰撞强度(collision strength)可选大小
Lambda=0
#带电粒子使用Coulombian case
gamma=-3
#总电荷数及质量(collision strength)可选大小
Q=2*v


gamma_decay=-1/(0.5)**3*np.power(np.pi/8,0.5)*np.exp(-1.5-0.5*0.5**-2)-Lambda*(2/(9*np.pi))**0.5
print(gamma_decay)

#保存到npz文件，且算相对误差
def save_and_draw(T,v,n,point,N,yipuxilong,dt):
    t = np.linspace(0, T, num=1 + int(T/Dt))
    Y = two_dimension(T=T, L=v, n=n, Lambda=Lambda, gamma=gamma)
    V, X,mass, momentum, energy1, energy2,electric_energy, entropy1, entropy2 = Y.solve(dt=dt,yipuxilong=yipuxilong, input=point)
    #np.save(f'T={T},L={int(v/np.pi)}pi,dt={dt},n={n},N={N},n0={n0},Lambda={Lambda},yipuxilong={yipuxilong}非齐次Landau damping,d=1D+2V=3 blob 连续拟合离散 mass', mass)
    #np.save(f'T={T},L={int(v/np.pi)}pi,dt={dt},n={n},N={N},n0={n0},Lambda={Lambda},yipuxilong={yipuxilong}非齐次Landau damping,d=1D+2V=3 blob 连续拟合离散 momentum', momentum)
    #np.save(f'T={T},L={int(v/np.pi)}pi,dt={dt},n={n},N={N},n0={n0},Lambda={Lambda},yipuxilong={yipuxilong}非齐次Landau damping,d=1D+2V=3 blob 连续拟合离散 energy', energy1)
    np.save(f'T={T},L={int(v/np.pi)}pi,dt={dt},n={n},N={N},n0={n0},Lambda={Lambda},α={alpha},yipuxilong={yipuxilong}非齐次Landau damping,d=1D+2V=3 blob 粒子 energy', energy2)
    np.save(f'T={T},L={int(v / np.pi)}pi,dt={dt},n={n},N={N},n0={n0},Lambda={Lambda},α={alpha},yipuxilong={yipuxilong}非齐次Landau damping,d=1D+2V=3 blob 电场 energy',electric_energy)
    #np.save(f'T={T},L={int(v/np.pi)}pi,dt={dt},n={n},N={N},n0={n0},Lambda={Lambda},yipuxilong={yipuxilong}非齐次Landau damping,d=1D+2V=3 blob 连续拟合离散 entropy', entropy1)
    np.save(f'T={T},L={int(v/np.pi)}pi,dt={dt},n={n},N={N},n0={n0},Lambda={Lambda},α={alpha},yipuxilong={yipuxilong}非齐次Landau damping,d=1D+2V=3 blob 粒子 entropy', entropy2)

    #plt.plot(t, mass)
    #plt.xlabel("T")
    #plt.title('Mass')
    #plt.hlines(y=mass[0], xmin=0, xmax=T, colors='r', linestyles='--')
    #plt.savefig(f'T={T},L={int(v/np.pi)}pi,dt={dt},n={n},N={N},n0={n0},Lambda={Lambda},yipuxilong={yipuxilong}非齐次Landau damping,d=1D+2V=3 blob 连续拟合离散 mass.eps', dpi=160)
    #plt.close()
    # plt.show()
    #plt.plot(t, momentum[:,0], label='momentum_x')
    #plt.plot(t, momentum[:,-1], label='momentum_y')
    #plt.legend(framealpha=1)
    #plt.xlabel("T")
    #plt.title('Momentum')
    #plt.savefig(f'T={T},L={int(v/np.pi)}pi,dt={dt},n={n},N={N},n0={n0},Lambda={Lambda},yipuxilong={yipuxilong}非齐次Landau damping,d=1D+2V=3 blob 连续拟合离散 momentum.eps', dpi=160)
    #plt.close()
    # plt.show()
    #plt.plot(t, energy1)
    #plt.xlabel("T")
    #plt.title('Kinetic energy')
    #plt.hlines(y=energy1[0], xmin=0, xmax=T, colors='r', linestyles='--')
    #plt.savefig(f'T={T},L={int(v/np.pi)}pi,dt={dt},n={n},N={N},n0={n0},Lambda={Lambda},yipuxilong={yipuxilong}非齐次Landau damping,d=1D+2V=3 blob 连续拟合离散 kinetic energy.eps', dpi=160)
    #plt.close()
    # plt.show()
    plt.plot(t, energy2)
    plt.xlabel("T")
    plt.title('Kinetic energy')
    plt.hlines(y=energy2[0], xmin=0, xmax=T, colors='r', linestyles='--')
    plt.savefig(f'T={T},L={int(v/np.pi)}pi,dt={dt},n={n},N={N},n0={n0},Lambda={Lambda},α={alpha},yipuxilong={yipuxilong}非齐次Landau damping,d=1D+2V=3 blob 离散 kinetic energy.png',bbox_inches='tight')
    plt.close()
    # plt.show()
    plt.plot(t, np.power(2*np.array(electric_energy),0.5),label='numerical result')
    plt.xlabel("T")
    plt.title('Electric field L2 norm')
    #plt.hlines(y=np.power(electric_energy[0]*2,0.5), xmin=0, xmax=T, colors='r', linestyles='--')
    plt.plot(t[:int(30/Dt)+1],np.power(electric_energy[0]*2,0.5)*np.exp(gamma_decay*t[:int(30/Dt)+1]),'--',label='the decay predicted by theory in the collisionless case')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'T={T},L={int(v / np.pi)}pi,dt={dt},n={n},N={N},n0={n0},Lambda={Lambda},α={alpha},yipuxilong={yipuxilong}非齐次Landau damping,d=1D+2V=3 blob 电场 energy.png',bbox_inches='tight')

    plt.close()
    # plt.show()
    plt.plot(t, np.array(electric_energy)+np.array(energy2))
    plt.xlabel("T")
    plt.title('Total energy')
    plt.hlines(y=(np.array(electric_energy)+np.array(energy2))[0], xmin=0, xmax=T, colors='r', linestyles='--')
    plt.savefig(f'T={T},L={int(v / np.pi)}pi,dt={dt},n={n},N={N},n0={n0},Lambda={Lambda},α={alpha},yipuxilong={yipuxilong}非齐次Landau damping,d=1D+2V=3 blob total energy.png',bbox_inches='tight')
    plt.close()
    # plt.show()
    #plt.plot(t, entropy1)
    #plt.xlabel("T")
    #plt.title('Entropy')
    #plt.savefig(f'T={T},L={int(v/np.pi)}pi,dt={dt},n={n},N={N},n0={n0},Lambda={Lambda},yipuxilong={yipuxilong}非齐次Landau damping,d=1D+2V=3 blob 连续拟合离散 entropy.eps', dpi=160)
    #plt.close()
    # plt.show()
    plt.plot(t, entropy2)
    plt.xlabel("T")
    plt.title('Entropy')
    plt.savefig(f'T={T},L={int(v/np.pi)}pi,dt={dt},n={n},N={N},n0={n0},Lambda={Lambda},α={alpha},yipuxilong={yipuxilong}非齐次Landau damping,d=1D+2V=3 blob 离散 entropy.eps',bbox_inches='tight')
    plt.close()
    # plt.show()

    print( 'ok')

save_and_draw(T=T,v=v,n=n,point=point,N=N,yipuxilong=yipuxilong,dt=dt)