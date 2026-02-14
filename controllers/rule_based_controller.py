import numpy as np

def rule_based_controller(
    net_demand,
    transformer_capacity,
    fixed_load=None,
    solar_generation=None,
    min_service_ratio=0.5
):
    """
    Advanced rule-based controller for smart grid balancing.
    """

    num_houses = len(net_demand)
    actions = np.ones(num_houses)

    total_demand = np.sum(net_demand)

    # ---------------------------
    # Rule 1: No overload
    # ---------------------------
    if total_demand <= transformer_capacity:
        return actions

    overload_ratio = total_demand / transformer_capacity

    # ---------------------------
    # Rule 2: Mild overload
    # ---------------------------
    if overload_ratio <= 1.2:

        # Protect fixed load if provided
        if fixed_load is not None:
            variable_part = np.maximum(net_demand - fixed_load, 0)

            total_variable = np.sum(variable_part)

            if total_variable > 0:
                allowed_variable = transformer_capacity - np.sum(fixed_load)
                scale_var = max(0.0, allowed_variable / total_variable)

                actions = (fixed_load + scale_var * variable_part) / net_demand

        else:
            scale = transformer_capacity / total_demand
            actions *= scale

    # ---------------------------
    # Rule 3: Severe overload
    # ---------------------------
    else:
        scale = transformer_capacity / total_demand
        actions = np.maximum(scale, min_service_ratio)

    # ---------------------------
    # Solar bonus
    # ---------------------------
    if solar_generation is not None:
        solar_bonus = solar_generation / (np.max(solar_generation) + 1e-6)
        actions += 0.05 * solar_bonus

    return np.clip(actions, 0.0, 1.0)
