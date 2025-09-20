from function import mollifier, square_collision_kernel_A, random_group
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(10)
import os
import random
random.seed(10)
import time


class three_dimension_Coulomb_EM_solution():
    def __init__(self,T:float,L:float,n:int,Lambda:float,gamma:float,epsilon:float,sigma:float=0.7):
        """Define the basic quantity of the algorithm.

        Parameters
        ----------
        T : float
            The upper bound of time.
        L : float
            The truncated value of the velocity component.
        n : int
            The number of points taken for [-L, L] (including both ends) then minus one.
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
        self.d = 3
        self.T = T
        self.L = L
        self.n = n
        self.Lambda = Lambda
        self.gamma = gamma
        self.a = np.linspace(-self.L, self.L, n + 1, dtype=float)
        self.h = 2 * L / n
        self.a_ = np.linspace(-self.L + self.h / 2, self.L - self.h / 2, n, dtype=float)
        Q_ = np.mgrid[-self.L + self.h / 2: self.L + self.h / 2: self.h,
            -self.L + self.h / 2: self.L + self.h / 2: self.h,
            -self.L + self.h / 2: self.L + self.h / 2: self.h]
        self.vx_ = Q_[0].reshape([-1, 1])
        self.vy_ = Q_[1].reshape([-1, 1])
        self.vz_ = Q_[2].reshape([-1, 1])
        self.V_ = np.concatenate((self.vx_, self.vy_, self.vz_), axis=1)
        self.reference_solution = np.load(os.path.join(os.getcwd(),\
            'RBM_reference_solution\\T=200_L=2_dt=0.05_n=40_epsilon=0.01_every_second_RBM_solution.npy'))
        self.n0=int(np.round(np.round(self.reference_solution.shape[1]/self.d)**(1/self.d),0))
        self.h0 = 2 * L / self.n0
        self.a__ = np.linspace(-self.L + self.h0 / 2, self.L - self.h0 / 2, self.n0, dtype=float)
        Q__ = np.mgrid[-self.L + self.h0 / 2: self.L + self.h0 / 2: self.h0,
            -self.L + self.h0 / 2: self.L + self.h0 / 2: self.h0,
            -self.L + self.h0 / 2: self.L + self.h0 / 2: self.h0]
        self.vx__ = Q__[0].reshape([-1, 1])
        self.vy__ = Q__[1].reshape([-1, 1])
        self.vz__ = Q__[2].reshape([-1, 1])
        self.V__ = np.concatenate((self.vx__, self.vy__, self.vz__), axis=1)

    def get_reference_solution(self, T:float)->np.ndarray:
        """Give the reference solution(RBM) at time T.

        Parameters
        ----------
        T : float
            Time.

        Returns
        -------
        Array : np.ndarray
            An velocity matrix of particles of the reference solution(RBM) at time T, whose shape is (n^3, 3).
        """

        if T!=0:
            return self.reference_solution[int(T-1),:].reshape(-1,int(self.d))
        else:
            return self.V__

    def compute_parameter(self,V:np.ndarray,num:int,T:float)->list:
        """Compute some statistic: kinetic energy, entropy, relative_L2_error and relative entropy.

        Parameters
        ----------
        V : np.ndarray
            The velocity matrix of particles at time T, whose shape is (num, 2).
        num : int
            Total number of particles.
        T : float
            Time when computing.

        Returns
        -------
        list
            A list contains kinetic energy, entropy, relative_L2_error and relative entropy.
        """
        
        kinetic_energy = 0.5 / num * np.sum(np.power(V, 2))
        
        Z = np.zeros(shape=[self.n ** self.d, 1])
        Z_reference = np.zeros(shape=[self.n ** self.d, 1])
        for i in range(self.n ** self.d):
            x = np.repeat(self.V_[i].reshape(1, -1), num, axis=0) - V
            z = np.linalg.norm(x, axis=-1)
            choice = np.where(z < self.sigma)
            x = x[choice]
            Z[i] = np.sum(mollifier(d=self.d, x=x, epsilon=epsilon), axis=0) / num

        V_reference = self.get_reference_solution(T=T)
        Z1 = np.concatenate((np.zeros(shape=[self.n0 ** int(self.d), 1]), 0.1 * np.ones(shape=[self.n0 ** int(self.d), 1]), np.zeros(shape=[self.n0 ** int(self.d), 1])), axis=1)
        Z2 = np.concatenate((-0.3 * np.ones(shape=[self.n0 ** int(self.d), 1]), 0.7 * np.ones(shape=[self.n0 ** int(self.d), 1]), 0.1 * np.ones(shape=[self.n0 ** int(self.d), 1])),axis=1)
        Z3 = np.concatenate((0.6 * np.ones(shape=[self.n0 ** int(self.d), 1]), -0.4 * np.ones(shape=[self.n0 ** int(self.d), 1]), -0.5 * np.ones(shape=[self.n0 ** int(self.d), 1])),axis=1)
        z_initial = 1 / (3 * (Standard_Deviation ** 2 * 2 * np.pi) ** 1.5) * (2.25 * np.exp(-np.linalg.norm(self.V__ - Z1, axis=-1) ** 2/
            (2 * Standard_Deviation ** 2)) + 0.25 * np.exp(-np.linalg.norm(self.V__ - Z2, axis=-1) ** 2 / (2 * Standard_Deviation ** 2)) + 0.5 * np.exp(
            -np.linalg.norm(self.V__ - Z3, axis=-1) ** 2 / (2 * Standard_Deviation ** 2))).reshape(-1, 1) * self.h0 ** self.d

        for i in range(self.n ** int(self.d)):
            x = np.repeat(self.V_[i].reshape(1, -1), self.n0 ** int(self.d), axis=0) - V_reference
            Z_reference[i] = np.sum(np.multiply(mollifier(d=self.d, x=x, epsilon=epsilon).reshape([-1, 1]), z_initial),axis=0)

        relative_L2_error = np.linalg.norm(Z - Z_reference) / np.linalg.norm(Z_reference)

        indices=np.where(Z>0)
        entropy = np.sum(np.multiply(Z[indices].ravel(), np.log(Z[indices]).ravel()))

        indices=np.where(np.multiply(Z,np.power(Z_reference,-1)).ravel()>0)
        relative_entropy = np.sum(np.multiply(Z.ravel()[indices], (np.log(np.multiply(Z.ravel()[indices],np.power(Z_reference.ravel()[indices],-1))))))

        return [kinetic_energy,relative_L2_error,entropy,relative_entropy]

    def solve(self,dt:float,V:np.ndarray)->tuple:
        """Solve Landau equation.

        Parameters
        ----------
        dt : float
            Time step.
        V : np.ndarray
            The velocity matrix of particles at time T, whose shape is (num, 2).

        Returns
        -------
        tuple
            A tuple contains velocity matrix of particles, kinetic energy, entropy, relative_L2_error, relative entropy and computaion time.
        """
        
        length=V.shape[0]

        kinetic_energy, relative_L2_error, entropy, relative_entropy = [], [], [], []
        
        parameter = self.compute_parameter(V=V, num=length, T=0)
        kinetic_energy.append(parameter[0])
        relative_L2_error.append(parameter[1])
        entropy.append(parameter[2])
        relative_entropy.append(parameter[3])

        V_temporary = np.zeros_like(V)
        Time = 0
        for t in range(int(self.T / dt)):
            time_start = time.time()
            group=random_group(list(range(length)),group_size=2)
            for i in range(int(length/2)):
                p1=group[i,0]
                p2=group[i,-1]
                V1=V[p1].reshape(1, -1)
                V2=V[p2].reshape(1, -1)
                z=V1-V2
                brown = np.random.normal(loc=0, scale=dt ** 0.5, size=[self.d, 1])
                Dv1 = -self.Lambda * z * (self.d - 1) * dt + np.matmul(square_collision_kernel_A(Lambda=self.Lambda, d=self.d, x=z,
                    gamma=self.gamma), brown).reshape(1,-1)
                V_temporary[p1] = V1 + Dv1.ravel()
                V_temporary[p2] = V2 - Dv1.ravel()
            V = V_temporary
            time_end = time.time()
            Time=Time+time_end-time_start

            if t%DT==(DT-1):
                parameter = self.compute_parameter(V=V, num=length, T=(t + 1) * dt)
                kinetic_energy.append(parameter[0])
                relative_L2_error.append(parameter[1])
                entropy.append(parameter[2])
                relative_entropy.append(parameter[3])
                print(parameter[0])

            print(f't={np.round((t + 1) * dt, 3)} finished')

            if t % int(1 / dt) == int(1 / dt) - 1:
                np.save(os.path.join(address,f'T={(t + 1) * dt}_velocity'), V)

        return V,kinetic_energy,relative_L2_error,self.h**self.d*np.array(entropy),self.h**self.d*np.array(relative_entropy),Time


T=200
v=2
n=30
n0=40
dt=0.1
Dt=1
point=np.load(os.path.join(os.getcwd(),'initial_distribution_sampling\\500000_points_3d-Coulomb.npy'))[:10000]
N=point.shape[0]
DT=int(Dt/dt)
epsilon=0.01
Lambda=1/(4*np.pi)
Lambda=Lambda*2
gamma=-3
#* initital distribution standard deviation
Standard_Deviation=0.5
address=os.path.join(os.getcwd(),'3d_Coulomb','EM',f'T={T}_dt={dt}_n={n}_N={N}_epsilon={epsilon}')
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
    Y = three_dimension_Coulomb_EM_solution(T=T, L=v, n=n, Lambda=Lambda, gamma=gamma,epsilon=epsilon)
    V, kinetic_energy, relative_L2_error, entropy, relative_entropy, Time = Y.solve(dt=dt,V=point)

    np.save(os.path.join(address,f'particle-energy'), kinetic_energy)
    np.save(os.path.join(address,f'relative-L2-error'), relative_L2_error)
    np.save(os.path.join(address,f'entropy'), entropy)
    np.save(os.path.join(address,f'relative-entropy(EM_vs_reference)'),relative_entropy)
    np.save(os.path.join(address,f'total-time'), Time)

    plt.plot(t, kinetic_energy)
    plt.xlabel("T")
    plt.ylabel('Energy')
    plt.title('Energy')
    plt.hlines(y=kinetic_energy[0], xmin=0, xmax=T, colors='r', linestyles='--')
    plt.savefig(os.path.join(address,f'particle-energy.png'), dpi=160)
    plt.close()
    # plt.show()
    plt.plot(t, relative_L2_error)
    plt.xlabel("T")
    plt.ylabel('relative L2 error')
    plt.title('relative L2 error')
    plt.savefig(os.path.join(address,f'relative-L2-error.png'), dpi=160)
    plt.close()
    # plt.show()
    plt.plot(t, entropy)
    plt.xlabel("T")
    plt.ylabel('entropy')
    plt.title('entropy')
    plt.savefig(os.path.join(address,f'entropy.png'), dpi=160)
    plt.close()
    # plt.show()
    plt.plot(t, relative_entropy)
    plt.xlabel("T")
    plt.ylabel('relative entropy')
    plt.title('relative entropy')
    plt.savefig(os.path.join(address,f'relative-entropy(EM_vs_reference).png'),dpi=160)
    plt.close()
    # plt.show()

    print('ok')

save_and_draw(T=T,v=v,n=n,point=point,epsilon=epsilon,dt=dt)


