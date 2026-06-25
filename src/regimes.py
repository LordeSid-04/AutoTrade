"""
Centralized definition of market regimes for evaluation and data exclusion.
"""

# 8 Distinct Regimes for Walk-Forward Testing and RL Data Exclusion (5 Validation, 3 Test)
REGIMES = {
    # --- Validation Regimes ---
    "Val: Post-Crisis (2011-2012)": ("2011-01-01", "2012-12-31"),
    "Val: Taper Tantrum (2013-2014)": ("2013-01-01", "2014-12-31"),
    "Val: Rate Normalization (2015-2016)": ("2015-01-01", "2016-12-31"),
    "Val: Trade War Choppy (2018-2019)": ("2018-01-01", "2019-12-31"),
    "Val: Tech Recovery (2023-2024)": ("2023-01-01", "2024-12-31"),
    # --- Hold-out Test Regimes ---
    "Test: Quiet Bull (2017)": ("2017-01-01", "2017-12-31"),
    "Test: COVID Crash (2020)": ("2020-01-01", "2020-12-31"),
    "Test: Rate Hike Choppy (2022)": ("2022-01-01", "2022-12-31")
}
