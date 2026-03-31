import numpy as np

def body_lin_vel_l2(
    vel: tuple[np.float32, np.float32],
    x: bool = True,
    y: bool = True
) -> np.float32:
    """
    Body linear velocity L2 norm
    Can choose which directions to include
    Returns a non-negative value (the norm)
    """
    
    x_vel = vel[0] if x else 0
    y_vel = vel[1] if y else 0
    
    return (x_vel ** 2 + y_vel ** 2) ** 0.5

def leg_contact(
    contacts: tuple[np.float32, np.float32],
    leg_1: bool = True,
    leg_2: bool = True,
) -> np.float32:
    """
    Leg contact reward / penalty
    Can choose which legs to include
    Returns a non-negative value (0 if no contact, 1 if specified has contact)
    """
    
    return np.min(np.array([contacts[0] * leg_1 + contacts[1] * leg_2, 1]))