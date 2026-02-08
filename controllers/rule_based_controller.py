import numpy as np


def rule_based_controller(net_demand, transformer_capacity):
    """
    Proportional load shedding controller.

    Parameters:
        net_demand (np.ndarray): Net demand per house (kW)
        transformer_capacity (float): Transformer capacity (kW)

    Returns:
        actions (np.ndarray): Action per house in [0, 1]
    """

    total_demand = np.sum(net_demand)

    # No overload → serve everything
    if total_demand <= transformer_capacity:
        return np.ones_like(net_demand)

    # Overload → proportional shedding
    scaling_factor = transformer_capacity / total_demand

    actions = scaling_factor * np.ones_like(net_demand)

    return np.clip(actions, 0.0, 1.0)
