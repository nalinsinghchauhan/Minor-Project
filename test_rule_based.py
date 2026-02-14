import numpy as np
import config

from env.smart_rules_grid_env import SmartGridEnv
from controllers.rule_based_controller import rule_based_controller

np.set_printoptions(precision=3, suppress=True)

env = SmartGridEnv(config)
state = env.reset()
done = False

print("\nRunning Rule-Based Controller (Advanced)\n")

while not done:

    actions = rule_based_controller(
        net_demand=state,
        transformer_capacity=config.TRANSFORMER_CAPACITY,
        fixed_load=env.fixed_load,
        solar_generation=env.solar_generation
    )

    next_state, reward, done, info = env.step(actions)

    print("\n--- Step ---")
    print("Demand BEFORE:  ", info["demand_before"])
    print("Actions:        ", info["actions"])
    print("Load served:    ", info["actual_load"])
    print("Demand AFTER:   ", info["demand_after"])
    print("Total load:     ", round(info["total_load"], 2))
    print("Overload:       ", round(info["overload"], 2))
    print("Reward:         ", round(reward, 2))

    state = next_state

print("\nEpisode finished.")
