import numpy as np


class SmartGridEnv:
    def __init__(
        self,
        num_houses=5,
        episode_length=24,
        transformer_capacity=15.0,
        fixed_load_range=(0.5, 1.5),
        variable_load_range=(0.0, 4.0),
        solar_houses=None,
        max_solar_generation=3.0
    ):
        self.num_houses = num_houses
        self.episode_length = episode_length
        self.transformer_capacity = transformer_capacity

        self.fixed_load_range = fixed_load_range
        self.variable_load_range = variable_load_range

        self.solar_houses = solar_houses or []
        self.max_solar_generation = max_solar_generation

        self.current_step = 0

        self.fixed_load = None
        self.variable_load = None
        self.solar_generation = None
        self.net_demand = None

        self.reset()

    def reset(self):
        self.current_step = 0

        # Fixed load (same every timestep)
        self.fixed_load = np.random.uniform(
            self.fixed_load_range[0],
            self.fixed_load_range[1],
            self.num_houses
        )

        self._generate_variable_and_solar()
        return self._get_state()

    def _generate_variable_and_solar(self):
        # Variable load (changes every timestep)
        self.variable_load = np.random.uniform(
            self.variable_load_range[0],
            self.variable_load_range[1],
            self.num_houses
        )

        # Solar generation
        self.solar_generation = np.zeros(self.num_houses)

        for i in self.solar_houses:
            self.solar_generation[i] = np.random.uniform(
                0.0,
                self.max_solar_generation
            )

        # Net demand cannot be negative
        self.net_demand = np.maximum(
            self.fixed_load + self.variable_load - self.solar_generation,
            0.0
        )

    def step(self, actions):
        actions = np.clip(actions, 0.0, 1.0)

        actual_load = actions * self.net_demand
        total_load = np.sum(actual_load)

        overload = max(0.0, total_load - self.transformer_capacity)

        reward = np.sum(actual_load) - 2.0 * overload

        self.current_step += 1
        done = self.current_step >= self.episode_length

        self._generate_variable_and_solar()

        info = {
            "total_load": total_load,
            "overload": overload,
            "fixed_load": self.fixed_load.copy(),
            "variable_load": self.variable_load.copy(),
            "solar_generation": self.solar_generation.copy(),
            "net_demand": self.net_demand.copy()
        }

        return self._get_state(), reward, done, info

    def _get_state(self):
        return self.net_demand.copy()
