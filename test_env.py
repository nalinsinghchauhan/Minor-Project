import numpy as np
from env.smart_grid_env import SmartGridEnv

# 🔧 CHANGE NUMBER OF HOUSES HERE
NUM_HOUSES = 12

np.set_printoptions(precision=3, suppress=True)

# 🔧 HOUSES WITH SOLAR (index-based)
SOLAR_HOUSES = [1, 4]

env = SmartGridEnv(
    num_houses=NUM_HOUSES,
    transformer_capacity=15.0,
    solar_houses=SOLAR_HOUSES,
    max_solar_generation=3.0
)

state = env.reset()
done = False

print("Solar houses:", SOLAR_HOUSES)
print("Initial net demand:", state)

while not done:
    actions = np.random.rand(env.num_houses)

    state, reward, done, info = env.step(actions)

    print("\n--- Step ---")
    print("Fixed load:     ", info["fixed_load"])
    print("Variable load:  ", info["variable_load"])
    print("Solar generation", info["solar_generation"])
    print("Net demand:     ", info["net_demand"])
    print("Total load:", round(info["total_load"], 3))
    print("Overload:", round(info["overload"], 3))
    print("Reward:", round(reward, 3))
