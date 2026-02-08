import numpy as np
import config
from env.smart_grid_env import SmartGridEnv
from controllers.rule_based_controller import rule_based_controller

np.set_printoptions(precision=3, suppress=True)

env = SmartGridEnv(config)

state = env.reset()
done = False

total_reward = 0.0
total_overload = 0.0

print("\nRunning Rule-Based Controller\n")

while not done:
    actions = rule_based_controller(
        net_demand=state,
        transformer_capacity=config.TRANSFORMER_CAPACITY
    )

    state, reward, done, info = env.step(actions)

    total_reward += reward
    total_overload += info["overload"]

    print("\n--- Step ---")
    print("Net demand:   ", info["net_demand"])
    print("Actions:      ", actions)
    print("Total load:   ", round(info["total_load"], 2))
    print("Overload:     ", round(info["overload"], 2))
    print("Reward:       ", round(reward, 2))

print("\nEpisode finished.")
print("Total reward: ", round(total_reward, 2))
print("Total overload:", round(total_overload, 2))
