import matplotlib.pyplot as plt
import numpy as np
np.random.seed(10)
import os
import random
random.seed(10)


def target_density_function(vx:float, vy:float)->float:
    """Compute the target density function.

    Parameters
    ----------
    vx : float
        velocity at x,.
    vy : float
        velocity at y.

    Returns
    -------
    float
        The value of the target density function at (vx, vy).
    """
    
    return 1 / np.pi * np.exp(-vx ** 2 - vy ** 2) * (vx ** 2 + vy ** 2)

def proposal_distribution(vx:float, vy:float)->float:
    """Sample the proposal distribution function.

    Parameters
    ----------
    vx : float
        velocity at x.
    vy : float
        velocity at y.

    Returns
    -------
    float
        The sample of the proposal distribution function with parameter (vx, vy).
    """
    
    return np.random.multivariate_normal(mean=np.array([vx, vy]), cov=np.eye(2))

def proposal_density_function(vx:float, vy:float, mean1:float, mean2:float)->float:
    """Compute the proposal density function.
    Parameters
    ----------
    vx : float
        velocity at x.
    vy : float
        velocity at y.
    mean1 : float
        parameter at x.
    mean2 : float
        parameter at y.

    Returns
    -------
    float
        The value of the  proposal density function at (vx, vy) with parameter (mean1, mean2).
    """
    
    return 1 / (2 * np.pi) * np.exp((-(vx - mean1) ** 2 - (vy - mean2) ** 2) / 2)

def metropolis_hastings(initial_data:list, n_samples:int)->np.ndarray:
    """metropolis hastings algorithm sampling

    Parameters
    ----------
    initial_data : list
        initial sample.
    n_samples : int
        num of samples.

    Returns
    -------
    np.ndarray
        samples of target distribution.
    """
    
    current_data_x = initial_data[0]
    current_data_y = initial_data[1]
    accepted_samples = []
    for i in range(n_samples*5+5000):
        proposed_data = proposal_distribution(current_data_x, current_data_y).ravel()
        A = target_density_function(proposed_data[0], proposed_data[1]) * proposal_density_function(vx=current_data_x,vy=current_data_y,
            mean1=proposed_data[0],mean2=proposed_data[1])
        B = target_density_function(current_data_x, current_data_y) * proposal_density_function(vx=proposed_data[0],
            vy=proposed_data[1],mean1=current_data_x, mean2=current_data_y)
        r = np.minimum(A / B, 1)
        u = np.random.rand()
        if u <= r:
            current_data_x = proposed_data[0]
            current_data_y = proposed_data[1]
            accepted_samples.append(proposed_data)
        else:
            accepted_samples.append(np.array([current_data_x, current_data_y]))
        if i % 10000 == 0:
            print(i)

    return np.array(accepted_samples)[5000::5,]

N=100000
a=metropolis_hastings(initial_data=[-1.414,0],n_samples=N)
address=os.path.join(os.getcwd(),'initial_distribution_sampling',f'{N}_points_2d-Maxwell')

if not os.path.exists(os.path.join(os.getcwd(),'initial_distribution_sampling',f'{N}_points_2d-Maxwell.npy')):
    np.save(address,a)

