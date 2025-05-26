import matplotlib.pyplot as plt
import numpy as np
import random
np.random.seed(10)
random.seed(10)

N=20000
#n=400
#h=2*5/n
a=np.random.multivariate_normal(mean=[-2,1],cov=np.eye(2),size=[N])
b1=np.random.multivariate_normal(mean=[1,-1],cov=np.eye(2),size=[N])
b2=np.random.multivariate_normal(mean=[1,-1],cov=np.eye(2),size=[N])
b3=np.random.multivariate_normal(mean=[1,-1],cov=np.eye(2),size=[N])
b4=np.random.multivariate_normal(mean=[1,-1],cov=np.eye(2),size=[N])

#print(np.concatenate((a,b1,b2,b3,b4),axis=1))

A=np.concatenate((a,b1,b2,b3,b4),axis=1).reshape([-1,2])
#print(A)


np.save(f'100000_points_2d-Coulomb', A[0:100000,])

#plt.scatter(A[0:19000,][:,0],A[0:19000,][:,-1],s=0.15)
#plt.show()

#H,edge=np.histogramdd(A,bins=[np.linspace(-5,5,n+1),np.linspace(-5,5,n+1)])
#H=0.5*H/N


#X,Y=np.meshgrid(np.linspace(-5+h/2,5-h/2,n),np.linspace(-5+h/2,5-h/2,n))
#Z=np.concatenate((X.reshape(-1,1),Y.reshape(-1,1)),axis=1)
#Z0=np.concatenate((-2*np.ones(shape=[n**2,1]),1*np.ones(shape=[n**2,1])),axis=1)
#Z1=np.concatenate((-0*np.ones(shape=[n**2,1]),-1*np.ones(shape=[n**2,1])),axis=1)
#Q=1/(4*np.pi)*(np.exp(-np.linalg.norm(Z-Z0,axis=-1)**2/2)+np.exp(-np.linalg.norm(Z-Z1,axis=-1)**2/2)).reshape(-1,n)
#print(np.max(np.abs(H-Q*(5/n)**2)))