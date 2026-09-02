"""Reusable preprocessing functions for the training pipeline."""

import numpy as np

from src.config import INVALID_ZERO_COLUMNS


def replace_invalid_zeros(frame):
    """Convert clinically implausible zero measurements into missing values."""
    frame = frame.copy()
    frame[INVALID_ZERO_COLUMNS] = frame[INVALID_ZERO_COLUMNS].replace(0, np.nan)
    return frame
