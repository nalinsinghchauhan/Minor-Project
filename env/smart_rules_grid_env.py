import numpy as np


class SmartGridEnv:
    def __init__(self, config):
        self.num_houses = config.NUM_HOUSES
        self.episode_length = config.EPISODE_LENGTH
        self.transformer_capacity = config.TRANSFORMER_CAPACITY

        self.fixed_load_range = config.FIXED_LOAD_RANGE
        self.variable_load_range = config.VARIABLE_LOAD_RANGE

        self.solar_houses = config.SOLAR_HOUSES
        self.max_solar_generation = config.MAX_SOLAR_GENERATION

        self.overload_penalty = config.OVERLOAD_PENALTY

        self.current_step = 0

        self.fixed_load = None
        self.variable_load = None
        self.solar_generation = None
        self.net_demand = None

    def reset(self):
        self.current_step = 0

        self.fixed_load = np.random.uniform(
            self.fixed_load_range[0],
            self.fixed_load_range[1],
            self.num_houses
        )

        self._generate_demand()
        return self.net_demand.copy()

    def _generate_demand(self):
        self.variable_load = np.random.uniform(
            self.variable_load_range[0],
            self.variable_load_range[1],
            self.num_houses
        )

        self.solar_generation = np.zeros(self.num_houses)
        for i in self.solar_houses:
            self.solar_generation[i] = np.random.uniform(
                0.0, self.max_solar_generation
            )

        self.net_demand = np.maximum(
            self.fixed_load + self.variable_load - self.solar_generation,
            0.0
        )

    def step(self, actions):
        # Demand BEFORE action
        demand_before = self.net_demand.copy()

        actions = np.clip(actions, 0.0, 1.0)

        # Load actually served
        actual_load = actions * demand_before
        total_load = np.sum(actual_load)

        overload = max(0.0, total_load - self.transformer_capacity)

        reward = (
            total_load
            - self.overload_penalty * overload
        )

        # Move time forward
        self.current_step += 1
        done = self.current_step >= self.episode_length

        # Update demand for NEXT timestep
        self._generate_demand()

        demand_after = self.net_demand.copy()

        info = {
            "demand_before": demand_before,
            "actions": actions.copy(),
            "actual_load": actual_load.copy(),
            "demand_after": demand_after,
            "total_load": total_load,
            "overload": overload,
            "fixed_load": self.fixed_load.copy(),
            "variable_load": self.variable_load.copy(),
            "solar_generation": self.solar_generation.copy(),
        }

        return self._get_state(), reward, done, info

    def _get_state(self):
        return self.net_demand.copy()
