import matplotlib.pyplot as plt
import numpy as np
import random
np.random.seed(10)
random.seed(10)

N=50000
#初始分布标准差
sigma=0.5
#n=400
#h=2*5/n
#A=np.zeros(shape=)
#for i in range(9):
a=np.random.multivariate_normal(mean=[0,0.1,0],cov=sigma**2*np.eye(3),size=[N,9])
b=np.random.multivariate_normal(mean=[-0.3,0.7,0.1],cov=sigma**2*np.eye(3),size=[N,1])
c=np.random.multivariate_normal(mean=[0.6,-0.4,-0.5],cov=sigma**2*np.eye(3),size=[N,2])

A=np.concatenate((a,b,c),axis=1)
A=A.reshape([-1,3])
#print(A.shape)

np.save(f'500000个点(d=3库伦case,方差为{sigma**2})', A[0:500000,])

#H,edge=np.histogramdd(A,bins=[np.linspace(-5,5,n+1),np.linspace(-5,5,n+1)])
#H=0.5*H/N


#X,Y=np.meshgrid(np.linspace(-5+h/2,5-h/2,n),np.linspace(-5+h/2,5-h/2,n))
#Z=np.concatenate((X.reshape(-1,1),Y.reshape(-1,1)),axis=1)
#Z0=np.concatenate((-2*np.ones(shape=[n**2,1]),1*np.ones(shape=[n**2,1])),axis=1)
#Z1=np.concatenate((-0*np.ones(shape=[n**2,1]),-1*np.ones(shape=[n**2,1])),axis=1)
#Q=1/(4*np.pi)*(np.exp(-np.linalg.norm(Z-Z0,axis=-1)**2/2)+np.exp(-np.linalg.norm(Z-Z1,axis=-1)**2/2)).reshape(-1,n)
#print(np.max(np.abs(H-Q*(5/n)**2)))