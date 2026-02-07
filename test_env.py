import numpy as np
import config
from env.smart_grid_env import SmartGridEnv

np.set_printoptions(precision=3, suppress=True)

env = SmartGridEnv(config)

state = env.reset()
done = False

print("Number of houses:", config.NUM_HOUSES)
print("Solar houses:", config.SOLAR_HOUSES)
print("Initial net demand:", state)

while not done:
    actions = np.random.rand(env.num_houses)

    state, reward, done, info = env.step(actions)

    print("\n--- Step ---")
    print("Fixed load:    ", info["fixed_load"])
    print("Variable load: ", info["variable_load"])
    print("Net demand:    ", info["net_demand"])
    print("Solar:         ", info["solar_generation"])
    print("Total load:    ", round(info["total_load"], 2))
    print("Overload:      ", round(info["overload"], 2))
    print("Reward:        ", round(reward, 2))

print("\nEpisode finished.")
