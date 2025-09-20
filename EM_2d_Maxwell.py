from function import mollifier, square_collision_kernel_A, random_group
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(10)
import os
import random
random.seed(10)
import time


class two_dimension_Maxwell_EM_solution():
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
        
        #* self.V is the velocity grid point center matrix, whose shape is (n^2,2). 

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
        self.vx, self.vy = np.meshgrid(self.a, self.a)
        self.V = np.stack((self.vx, self.vy), axis=2).reshape(-1, self.d)

    def get_exact_solution(self, T:float)->np.ndarray:
        """Give the exact solution at time T.

        Parameters
        ----------
        T : float
            Time.

        Returns
        -------
        Array : np.ndarray
            A mass distribution matrix of particles of the exact solution at time T, whose shape is (n^2, 1).
        """

        K=1-np.exp(-T/8)/2
        z=np.linalg.norm(self.V,axis=-1)**2
        return (1/(2*np.pi*K)*np.exp(-z/(2*K))*(2-1/K+(1-K)/(2*K**2)*z)).reshape([-1,1])

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
        for i in range(self.n ** self.d):
            x = np.repeat(self.V[i].reshape(1, -1), num, axis=0) - V
            z = np.linalg.norm(x, axis=-1)
            choice = np.where(z < self.sigma)
            x = x[choice]
            Z[i] = np.sum(mollifier(self.d, x, self.epsilon), axis=0) / num
        
        Z_exact_solution = self.get_exact_solution(T=T)
    
        relative_L2_error = np.linalg.norm(Z - Z_exact_solution) / np.linalg.norm(Z_exact_solution)

        indices=np.where(Z>0)
        entropy = np.sum(np.multiply(Z[indices].ravel(), np.log(Z[indices]).ravel()))

        indices=np.where(np.multiply(Z,np.power(Z_exact_solution,-1)).ravel()>0)
        relative_entropy = np.sum(np.multiply(Z.ravel()[indices], (np.log(np.multiply(Z.ravel()[indices],np.power(Z_exact_solution.ravel()[indices],-1))))))

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
            group = random_group(list(range(length)), group_size=2)
            for i in range(int(length / 2)):
                p1 = group[i, 0]
                p2 = group[i, -1]
                V1 = V[p1].reshape(1, -1)
                V2 = V[p2].reshape(1, -1)
                z = V1 - V2
                brown = np.random.normal(loc=0, scale=dt ** 0.5, size=[self.d, 1])
                Dv1 = -self.Lambda * z * (self.d - 1) * dt + np.matmul(square_collision_kernel_A(Lambda=self.Lambda, d=self.d, x=z,
                    gamma=self.gamma),brown).T
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

            print(f't={np.round((t + 1) * dt, 3)} finished')

            if t % int(1 / dt) == int(1 / dt) - 1:
                np.save(os.path.join(address,f'T={(t + 1) * dt}_velocity'), V)

        return V,kinetic_energy,relative_L2_error,self.h**self.d*np.array(entropy),self.h**self.d*np.array(relative_entropy),Time


def plot(T:float,V:np.ndarray,L:float,n:int,dt:float,address:str)->None:
    """plot the 3d graph of solution at time T

    Parameters
    ----------
    T : float
        Time.
    V : np.ndarray
        The velocity matrix of particles at time T, whose shape is (num, 2).
    L : float
        The truncated value of the velocity component.
    n : int
        The number of points taken for [-L, L] (including both ends) then minus one.
    dt : float
        Time step.
    address : str
        the address of saved image.
    """
    
    d=2
    length = V.shape[0]
    Z = np.zeros(shape=[n**d, 1])
    a = np.linspace(-v, v, n, dtype=float)
    VX, VY = np.meshgrid(a, a)
    V_ = np.stack((VX, VY), axis=2).reshape([-1, d])
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
        ZZ[i] = np.sum(mollifier(d=d, x=x, epsilon=epsilon).reshape(-1, 1), axis=0) / length
    plt.plot(a,ZZ,label='EM')
    plt.plot(a,(1 / (2 * np.pi * K) * np.exp(-p2** 2 / (2 * K)) * (2 - 1 / K + (1 - K) / (2 * K ** 2) * p2** 2)).ravel(),label='exact')
    plt.title('cross-section')
    plt.xlabel('v_y')
    plt.ylabel('f')
    plt.legend()
    plt.savefig(os.path.join(address,f'cross-section.png'),dpi=160)
    plt.close()
    # plt.show()

    for i in range(n**d):
        x = np.repeat(V_[i].reshape(1, -1), length, axis=0) - V
        z = np.linalg.norm(x, axis=-1)
        choice = np.where(z < 0.7)
        x = x[choice]
        Z[i] = np.sum(mollifier(d=d, x=x, epsilon=epsilon).reshape(-1, 1), axis=0) / length
    Z = Z.reshape([-1, n])
    fig = plt.figure(figsize=(17, 8))
    ax = fig.add_subplot(1,2,1,projection='3d')
    ax.plot_surface(VX, VY, Z-Z_, cmap=plt.cm.winter, alpha=1)
    ax.set_xlabel('v_x', fontsize=15)
    ax.set_ylabel('v_y', fontsize=15)
    ax.set_zlabel('n', fontsize=15)
    ax.set_title('EM-exact')

    ay = fig.add_subplot(1, 2, 2, projection='3d')
    ay.plot_surface(VX, VY, Z, cmap=plt.cm.winter,alpha=1)
    ay.set_xlabel('v_x', fontsize=15)
    ay.set_ylabel('v_y', fontsize=15)
    ay.set_zlabel('n', fontsize=15)
    ay.set_title('EM')
    plt.savefig(os.path.join(address,f'3d-graph.png'),dpi=160)
    plt.close()
    # plt.show()


T=200
v=4
n=100
dt=0.2
Dt=1
point=np.load(os.path.join(os.getcwd(),'initial_distribution_sampling\\100000_points_2d-Maxwell.npy'))[:10000]
N=point.shape[0]
DT=int(Dt/dt)
epsilon=0.01
Lambda=1/16
Lambda=Lambda*2
gamma=0
address=os.path.join(os.getcwd(),'2d_Maxwell','EM',f'T={T}_dt={dt}_n={n}_N={N}_epsilon={epsilon}')
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
    Y = two_dimension_Maxwell_EM_solution(T=T, L=v, n=n, Lambda=Lambda, gamma=gamma,epsilon=epsilon)
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

    plot(T=T, V=V, L=v, n=n, dt=dt,address=address)

    print( 'ok')

save_and_draw(T=T,v=v,n=n,point=point,epsilon=epsilon,dt=dt)