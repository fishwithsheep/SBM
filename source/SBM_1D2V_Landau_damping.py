from function import mollifier, random_group, batch_simulate_circle_BM
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(10)
import os
import random
random.seed(10)
import time


def shape_function(x:np.ndarray,delta_x:float)->np.ndarray:
    """shape function of first order.

    Parameters
    ----------
    x : np.ndarray
        input array.
    delta_x : float
        the size(length actually) of each grid.

    Returns
    -------
    np.ndarray
        the output after shape function.
    """
    
    z=np.zeros_like(x)
    X=np.abs(x)
    indice=np.where(X<=delta_x)
    z[indice]=1-X[indice]/delta_x
    return z


class Landau_damping_1D2V_SBM_solution():
    def __init__(self,T:float,L:float,n:int,n0:int,Lambda:float,gamma:float,epsilon:float,sigma:float=0.7):
        """Define the basic quantity of the algorithm.

        Parameters
        ----------
        T : float
            The upper bound of time.
        L : float
            The truncated value of the velocity component. [0, 2*L] is the domain of the sapce. 
        n : int
            The number of points taken for [-L, L] (including both ends) then minus one.
        n0 : int
            The number of points taken for [-L, L] (including both ends) then minus one, used only for drawing velocity distribution in space graph.
        Lambda :float
            Collision strength, which is one parameter of Landau equation (since different forms of collision kernel 
            in the Landau equation may differ by one constant, it should be check whether Lambda here need to be multiply
            some constant like two).
        gamma : float
            Determine the category of collision (-d-1<gamma<1), which is one parameter of Landau equation.
        epsilon : float
            Parameter of mollification kernel (epsilon>0).
        sigma : float, optional
            The distance threshold for computing mollifying (reducing computation time and ensure accuracy) by default 0.7.
        """
        
        #* self.V is the velocity grid point center matrix, whose shape is (n^3,3). 
        
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

    
    def solve(self,dt:float,all:np.ndarray,m:int=5,draw:bool=True)->tuple:
        
        """Solve Landau equation.

        Parameters
        ----------
        dt : float
            Time step.
        all : np.ndarray
            The velocity and position matrix (phase matrix) of particles at time T, whose shape is (num, 3). The first and second column 
            is about velocity, and the third column is about position.
        m : int, optional
            iteration num.
        draw : bool, optional
            whether draw or not the velocity distribution graph.

        Returns
        -------
        tuple
            A tuple contains velocity and position matrix (phase matrix) of particles, kinetic energy, electric energy and computaion time.
        """
        
        V=all[:,0:2]
        X=all[:,2:]
        length=all.shape[0]

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

        #* rho is the charge density at grid center, fai is the electric potential at grid center.
        rho = (np.sum(z,axis=0)/ (length * self.h) - 0.5 / self.L)*self.total
        rho_hat = np.fft.fft(rho)
        fai_hat = np.zeros_like(rho_hat)
        for i in range(1, int((self.n + 1) / 2)):
            fai_hat[i] = rho_hat[i] / ((0.5*i) ** 2)
            fai_hat[-i] = rho_hat[-i] / ((0.5*i) ** 2)
        fai = np.fft.ifft(fai_hat).real
        E = -(np.roll(fai,-1) - np.roll(fai,1)) / (2 * self.h)
        time_end = time.time()
        Time = Time + time_end - time_start
        electric_energy=[np.linalg.norm(E)**2*self.h/2]

        for t in range(int(self.T / dt)):
            time_start = time.time()
            #* collision step
            X_belong_position=np.digitize(X.ravel(), np.linspace(0, 2 * self.L, self.n + 1))-1
            for j in range(self.n):
                indices = np.where(X_belong_position == j)
                V_tem=V[indices]
                #* before_length is the particle num of each grid
                before_length=V_tem.shape[0]
                extra=np.zeros(shape=2)
                #* randomly choose one particle
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
                        t_batch=np.array((2 * self.Lambda ** 0.5 * np.linalg.norm(z) ** (self.gamma / 2)) ** 2 * dt),d=2))
                    z__ = (q * np.linalg.norm(z) - z) / 2
                    V_tem[p1] = V1 + z__.ravel()
                    V_tem[p2] = V2 - z__.ravel()
                if before_length % 2!=0:
                    V_tem=np.concatenate((V_tem[:choice],extra.reshape([1,-1]),V_tem[choice:]),axis=0)
                V[indices]=V_tem

            #* advection step
            X_temporary=np.zeros_like(X)
            V_temporary=np.zeros_like(V)
            E_temporary=np.zeros_like(E)

            X_belong_position = np.digitize(X.ravel(), np.linspace(0, 2 * self.L, self.n + 1))-1
            E_each_particle = E[X_belong_position]
            #* give the guessed value pf advection
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

            #* periodic boundary condition make mod 2*L
            X[:,0]=np.copy(X_temporary[:,0])%(2*self.L)
            V[:,0]=np.copy(V_temporary[:,0])
            E=np.copy(E_temporary)

            all=np.concatenate((V,X),axis=1)

            time_end = time.time()
            Time = Time + time_end - time_start


            if t%DT==(DT-1):
                particle_energy.append(0.5 * self.total / length * np.linalg.norm(V) ** 2)
                electric_energy.append(np.linalg.norm(E) ** 2*self.h/2)

            print(f't={np.round((t + 1) * dt, 3)} finished')

            if t % int(1 / dt) == int(1 / dt) - 1:
                np.save(os.path.join(address,f'T={(t + 1) * dt}_phase'), all)

                if draw:
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

        return all,particle_energy, electric_energy,Time


    def drawing1(self,matrix,T,N):
        np.save(os.path.join(address,f'T={T}_v_x_distribution_in_space'),matrix)
        plt.imshow(matrix, cmap='jet', origin='lower', extent=[0, 2*self.L, -self.L, self.L])
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('v_x')
        plt.title('v_x distribution in space')
        #plt.show()
        plt.savefig(os.path.join(address,f'T={T}_v_x_distribution_in_space_graph.eps'),bbox_inches='tight',dpi=160)
        plt.close()

    def drawing2(self,matrix,T,N):
        np.save(os.path.join(address,f'T={T}_v_y_distribution_in_space'),matrix)
        plt.imshow(matrix, cmap='jet', origin='lower', extent=[0, 2*self.L, -self.L, self.L])
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('v_y')
        plt.title('v_y distribution in space')
        #plt.show()
        plt.savefig(os.path.join(address,f'T={T}_v_y_distribution_in_space_graph.eps'),bbox_inches='tight',dpi=160)
        plt.close()

    def drawing3(self,matrix,T,N):
        np.save(os.path.join(address,f'T={T}_v_module_distribution_in_space'),matrix)
        plt.imshow(matrix, cmap='jet', origin='lower', extent=[0, 2*self.L, 0, 2**0.5*self.L])
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('|v|')
        plt.title('|v| distribution in space')
        #plt.show()
        plt.savefig(os.path.join(address,f'T={T}_v_module_distribution_in_space_graph.eps'),bbox_inches='tight',dpi=160)
        plt.close()



T=50
v=2*np.pi
n=128
n0=200
dt=0.02
Dt=0.02
alpha=0.1
point=np.load(os.getcwd(),'data',f'initial_distribution_sampling\\500000_points_Landau_damping_1D_2V_alpha={alpha}.npy')[:500000]
N=point.shape[0]
DT=int(Dt/dt)
epsilon=0.01
Lambda=1
gamma=-2
gamma_decay=-1/(0.5)**3*np.power(np.pi/8,0.5)*np.exp(-1.5-0.5*0.5**-2)-Lambda*(2/(9*np.pi))**0.5
address=os.path.join(os.getcwd(),'Landau_damping_1D2V','SBM',f'T={T}_L={int(v / np.pi)}pi_dt={dt}_n={n}_n0={n0}_N={N}_Lambda={Lambda}_alpha={alpha}_epsilon={epsilon}')
os.makedirs(address,exist_ok=True)


def save_and_draw(T:float,v:float,n:int,point:np.ndarray,epsilon:float,dt:float)->None:
    """ Solve the Landau equation and save data draw graph.

    Parameters
    ----------
    T : float
        Total time.
    v : float
        The truncated value of the velocity component.
    n : int
        The number of points taken for [-L, L] (including both ends) then minus one.
    point : np.ndarray
        The velocity matrix of particles at time T, whose shape is (num, 2).
    epsilon : float
        Parameter of mollification kernel (epsilon>0).
    dt : float
        Time step.
    """
    
    t = np.linspace(0, T, num=1 + int(T/Dt))
    Y = Landau_damping_1D2V_SBM_solution(T=T, L=v, n=n, n0=n0, Lambda=Lambda, gamma=gamma,epsilon=epsilon)
    all ,particle_energy ,electric_energy, Time= Y.solve(dt=dt,all=point,draw=False)
    np.save(os.path.join(address,f'particle-energy'), particle_energy)
    np.save(os.path.join(address,f'electric-energy'), electric_energy)
    np.save(os.path.join(address,f'total-time'), Time)

    plt.plot(t, particle_energy)
    plt.xlabel("T")
    plt.title('Kinetic energy')
    #plt.show()
    plt.savefig(os.path.join(address,f'kinetic-energy.png'), dpi=160)
    plt.close()

    plt.plot(t, np.power(2*np.array(electric_energy),0.5))
    plt.xlabel("T")
    plt.title('Electric field L_2 norm')
    #end=np.minimum(T,np.log(0.1)/gamma_decay)
    #plt.plot(t[:int(end/Dt)+1],np.power(2*electric_energy[0],0.5)*np.exp(gamma_decay*t[:int(end/Dt)+1]),'--',label='the decay predicted by theory')
    plt.yscale('log')
    #plt.legend()
    # plt.show()
    plt.savefig(os.path.join(address,f'Electric-field-L_2-norm.png'), dpi=160)
    plt.close()

    plt.plot(t, np.array(electric_energy)+np.array(particle_energy))
    plt.xlabel("T")
    plt.title('Total energy')
    plt.hlines(y=(np.array(electric_energy)+np.array(particle_energy))[0], xmin=0, xmax=T, colors='r', linestyles='--')
    # plt.show()
    plt.savefig(os.path.join(address,f'total-energy.png'), dpi=160)
    plt.close()

    print( 'ok')

save_and_draw(T=T,v=v,n=n,point=point,epsilon=epsilon,dt=dt)