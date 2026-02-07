# =========================
# GRID CONFIGURATION
# =========================

NUM_HOUSES = 10
EPISODE_LENGTH = 24

TRANSFORMER_CAPACITY = 15.0  # kW (realistic for 10 houses)

# =========================
# LOAD CONFIGURATION
# =========================

FIXED_LOAD_RANGE = (0.8, 1.5)      # kW
VARIABLE_LOAD_RANGE = (0.0, 4.0)   # kW

# =========================
# SOLAR CONFIGURATION
# =========================

SOLAR_HOUSES = [2, 5, 7]            # index-based
MAX_SOLAR_GENERATION = 3.0          # kW

# =========================
# REWARD CONFIGURATION
# =========================

OVERLOAD_PENALTY = 2.0
