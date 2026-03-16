"""
Configuration file for the Bayesian LOOCV pozzolan reactivity model.

Centralizes file paths, feature selection, and global parameters.
Features are commented/uncommented depending on the target variable.
"""

import os

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
if IS_COLAB:
    PATH_EXCEL = '/content/drive/MyDrive/Paper_NPz/Natural pozzolans 34 (data for the model)-3.xlsx'
    OUTPUT_DIR = '/content/drive/MyDrive/Paper_NPz/output'
else:
    # Local paths (relative to project root)
    PATH_EXCEL = os.path.join('data', 'Natural pozzolans 34 (data for the model)-3.xlsx')
    OUTPUT_DIR = 'output'

os.makedirs(OUTPUT_DIR, exist_ok=True)

PATH_CLEANED_DATA = os.path.join(OUTPUT_DIR, '01_cleaned_dataset.xlsx')
PATH_LOOCV_BAYES  = os.path.join(OUTPUT_DIR, 'BR_loocv_results.xlsx')

# ---------------------------------------------------------------------------
# Feature selection
# Uncomment/comment features according to the target variable being predicted.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    # 'SSA',                   # Specific Surface Area
    'D50',                     # Median particle size
    # 'Amorphous SiO2',       # Amorphous silica content
    # 'Amorphous Al2O3',      # Amorphous alumina content
    'Reactive Al2O3 (%)',      # Reactive alumina content
    'Reactive SiO2 (%)',       # Reactive silica content
    # 'Reactive phases (%)',   # Total reactive phases
    # 'Al2O3(%)',
    # 'SiO2 (%)',
    # 'Amorphous phases (%)',
]

TARGET_COLUMN = 'Heat release 7d'
ID_COLUMN = 'ID'

# ---------------------------------------------------------------------------
# Global parameters
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
