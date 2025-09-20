import matplotlib.pyplot as plt
import numpy as np
np.random.seed(10)
import os
import random
random.seed(10)


N=20000
a=np.random.multivariate_normal(mean=[-2,1],cov=np.eye(2),size=[N])
b1=np.random.multivariate_normal(mean=[1,-1],cov=np.eye(2),size=[N])
b2=np.random.multivariate_normal(mean=[1,-1],cov=np.eye(2),size=[N])
b3=np.random.multivariate_normal(mean=[1,-1],cov=np.eye(2),size=[N])
b4=np.random.multivariate_normal(mean=[1,-1],cov=np.eye(2),size=[N])


A=np.concatenate((a,b1,b2,b3,b4),axis=1).reshape([-1,2])

address=os.path.join(os.getcwd(),'initial_distribution_sampling',f'{5*N}_points_2d-Coulomb')

if not os.path.exists(os.path.join(os.getcwd(),'initial_distribution_sampling',f'{5*N}_points_2d-Coulomb.npy')):
    np.save(address,A[0:100000,])
