import matplotlib.pyplot as plt
import numpy as np
np.random.seed(10)
import os
import random
random.seed(10)


N=50000
#* initital distribution standard deviation
Standard_Deviation=0.5
a=np.random.multivariate_normal(mean=[0,0.1,0],cov=Standard_Deviation**2*np.eye(3),size=[N,9])
b=np.random.multivariate_normal(mean=[-0.3,0.7,0.1],cov=Standard_Deviation**2*np.eye(3),size=[N,1])
c=np.random.multivariate_normal(mean=[0.6,-0.4,-0.5],cov=Standard_Deviation**2*np.eye(3),size=[N,2])

A=np.concatenate((a,b,c),axis=1)
A=A.reshape([-1,3])

address=os.path.join(os.getcwd(),'initial_distribution_sampling',f'{10*N}_points_3d-Coulomb')

if not os.path.exists(os.path.join(os.getcwd(),'initial_distribution_sampling',f'{10*N}_points_3d-Coulomb.npy')):
    np.save(address,A[0:500000,])
