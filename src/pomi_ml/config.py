# -*- coding: utf-8 -*-
"""Project-level constants for the POMI machine-learning analysis."""

from __future__ import annotations

import numpy as np


RANDOM_STATE = 42
TRAIN_FRACTION = 0.70
N_BOOTSTRAP = 1000
RF_TREES = 100
RF_MIN_SAMPLES_LEAF = 5

TARGET = "Satisfaction_Overall"
FEATURES = [
    "Satisfaction_Access",
    "Satisfaction_Comfort",
    "Satisfaction_Cleanliness",
    "Satisfaction_Safety",
    "Satisfaction_Greenery",
    "Satisfaction_Maintenance",
    "Facility_Seating",
    "Facility_Fitness",
    "Facility_Children",
    "Facility_Shade",
    "Facility_WiFi",
]
FEATURE_LABELS = [
    "Access",
    "Comfort",
    "Clean",
    "Safety",
    "Green",
    "Maintain",
    "Seating",
    "Fitness",
    "Children",
    "Shade",
    "Wi-Fi",
]
MODEL_LABELS = ["Linear Reg", "SVR", "Random Forest", "GPR"]

PLOT_COLORS = np.array(
    [
        [0.8353, 0.2431, 0.3098],
        [0.9725, 0.6706, 0.3804],
        [0.4784, 0.6784, 0.8039],
        [0.2000, 0.2627, 0.5490],
        [0.8980, 0.5882, 0.5765],
        [0.9882, 0.8431, 0.6863],
        [0.7373, 0.8353, 0.9059],
        [0.6000, 0.6000, 0.7843],
        [0.1333, 0.5451, 0.1333],
        [0.8039, 0.5216, 0.2471],
        [0.8588, 0.4039, 0.4667],
    ]
)
