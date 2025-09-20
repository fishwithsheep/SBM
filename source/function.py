import matplotlib.pyplot as plt
import numpy as np
np.random.seed(10)
import os
import random
random.seed(10)
import time


def mollifier(d:int,x:np.ndarray,epsilon:float)->np.ndarray:
    """Mollify the input ndarray x with parameter epsilon.

    Parameters
    ----------
    d : int
        Dimension of Independent Variable(d>=2).
    x : np.ndarray
        A 2D numpy array, whose shape is (num, d).
    epsilon : float
        Parameter of mollification kernel (epsilon>0).

    Returns
    -------
    Array : np.ndarray
        A 2D numpy array after mollified, whose shape is (num, 1).
    """
    
    return (2*np.pi*epsilon)**(-d/2)*np.exp(-np.linalg.norm(x,axis=-1)**2/(2*epsilon)).reshape(-1,1)

def gradient_mollifier(d:int,x:np.ndarray,epsilon:float)->np.ndarray:
    """Give the gradient of mollifing the input ndarray x with parameter epsilon.

    Parameters
    ----------
    d : int
        Dimension of Independent Variable (d>=2).
    x : np.ndarray
        A 2D numpy array, whose shape is (num, d).
    epsilon : float
        Parameter of mollification kernel (epsilon>0).

    Returns
    -------
    Array : np.ndarray
        A 2D numpy array of gradient, whose shape is (num, 1).
    """
    
    return (2*np.pi*epsilon)**(-d/2)*np.multiply(np.exp(-np.linalg.norm(x,axis=-1)**2/(2*epsilon)).reshape(-1,1),(-x/epsilon))

def collision_kernel_A(Lambda:float,d:int,x:np.ndarray,gamma:float)->np.ndarray:
    """Give the collision kernel A (including compute a projection matrix).

    Parameters
    ----------
    Lambda : float
        Collision strength (Lambda>=0, when Lambda=0, actually no collision exists; when Lambda=+∞, it comes to the hydrodynamics limit).
    d : int
        Dimension of Independent Variable (d>=2).
    x : np.ndarray
        A 2D numpy array, whose shape is (1, d).
    gamma : float
        Determine the category of collision (-d-1<gamma<1), gamma=0 corresponds to the Maxwellian molecule.

    Returns
    -------
    Array : np.ndarray
        A 2D numpy array of the collision kernel A, whose shape is (num, 1).
    """
    
    a=np.linalg.norm(x,axis=-1)**2*np.eye(d)
    b = np.matmul(x.T,x)
    return Lambda * (np.linalg.norm(x, axis=-1) ** gamma) * (a - b)

def square_collision_kernel_A(Lambda:float,d:int,x:np.ndarray,gamma:float)->np.ndarray:
    """Give the square-root of the collision kernel A(using the property of the projection matrix).

    Parameters
    ----------
    Lambda : float
        Collision strength (Lambda>=0, when Lambda=0, actually no collision exists; when Lambda=+∞, it comes to the hydrodynamics limit).
    d : int
        Dimension of Independent Variable(d>=2).
    x : np.ndarray
        A 2D numpy array, whose shape is (1, d).
    gamma : float
        Determine the category of collision (-d-1<gamma<1), gamma=0 corresponds to the Maxwellian molecule.

    Returns
    -------
    Array : np.ndarray
        A 2D numpy array of the collision kernel A, whose shape is (num,1).
    """
    
    a = np.linalg.norm(x, axis=-1) ** 2 * np.eye(d)
    b = np.matmul(x.T, x)
    return np.power(Lambda,0.5) * (np.linalg.norm(x, axis=-1) ** (-1+gamma/2)) * (a - b)

def random_group(data:list, group_size:int=2)->np.ndarray:
    """Shuffle the order.

    Parameters
    ----------
    data : list
        A list.
    group_size : int, optional
        The num of each group, by default 2.

    Returns
    -------
    Array : np.ndarray
        A 2D numpy array, whose shape is (num//2, 2).
    """
    
    random.shuffle(data)
    num_groups = len(data) // group_size
    groups = []

    for i in range(num_groups):
        start_index = i * group_size
        end_index = (i + 1) * group_size if i != num_groups - 1 else len(data)
        group = data[start_index:end_index]
        groups.append(group)
    return np.reshape(groups,(-1,2))

def batch_simulate_circle_BM(z_batch:np.ndarray, t_batch:np.ndarray,d:int)->np.ndarray:
    """Simulate the Spherical Brownian motion with given time interval when d=2.

    Parameters
    ----------
    z_batch : np.ndarray
        A 2D numpy array, whose shape is (num, d). It is the start of the Spherical Brownian motion.
    t_batch : np.ndarray
        A 2D numpy array, whose shape is (num, 1). It is the time interval.
    d : int
        dimension of the full space, not the dimesnion of the sphere.

    Returns
    -------
    Array : np.ndarray
        A 2D numpy array, whose shape is (num, d). It is the end of the Spherical Brownian motion.
    """
    
    if d==2:
        batch_size, d = z_batch.shape
        angles = np.arctan2(z_batch[:, 1], z_batch[:, 0]).reshape(batch_size, 1)

        noise_mean = np.zeros([batch_size, 1]) 
        noise_std = np.sqrt(t_batch) 
        noise = np.random.normal(noise_mean, noise_std)
        noisy_angles = angles + noise

        noisy_z_batch = np.concatenate((np.cos(noisy_angles), np.sin(noisy_angles)), axis=1)
        return noisy_z_batch.reshape(batch_size, d)
    else:
        1

def sphere_brown(d:int,t:float,z:np.ndarray)->np.ndarray:
    """Simulate the Spherical Brownian motion with given time interval when d>2.

    Parameters
    ----------
    d : int
        dimension of the full space, not the dimesnion of the sphere.
    t : float
        time interval.
    z : np.ndarray
        A 2D numpy array, whose shape is (1, d). It is the start of the Spherical Brownian motion.

    Returns
    -------
    np.ndarray
        A 2D numpy array, whose shape is (1, d). It is the end of the Spherical Brownian motion.
    """
    
    x = 0
    W = [np.array(z)]
    ed = np.zeros(shape=[d])
    ed[-1] = 1
    p = z
    theta1 = (d - 1) / 2
    theta2 = (d - 1) / 2
    theta = theta1 + theta2
    beta = 0.5 * (theta - 1) * t
    if beta == 0:
        eta = 1
    else:
        eta = beta / (np.exp(beta) - 1)

    mu = 2 * eta / t

    if beta == 0:
        sigma = np.power(2 / (3 * t), 0.5)
    else:
        sigma = np.power(2 * eta / t * (eta + beta) ** 2 * (1 + eta / (eta + beta) - 2 * eta) * beta ** -2, 0.5)

    A_infty = np.random.normal(loc=mu, scale=sigma)
    while A_infty <= 0:
        A_infty = np.random.normal(loc=mu, scale=sigma)

    A_infty = np.round(A_infty)
    L = np.random.binomial(n=A_infty, p=x)
    X = np.random.beta(a=theta1 + L, b=theta2 + A_infty - L)

    Y = np.random.normal(size=[d - 1])
    Y = Y / np.linalg.norm(Y)

    u = ed - p.ravel()
    u = u.reshape([d, 1]) / np.linalg.norm(u)

    O = np.diag(np.ones(shape=[d])) - 2 * np.matmul(u, u.T)
    Z = np.zeros(shape=[d])
    Z[0:-1] = 2 * np.power(X * (1 - X), 0.5) * Y
    Z[-1] = 1 - 2 * X
    p = np.matmul(O, Z.reshape([-1, 1])).ravel()
    W.append(p)
    return p