"""Tests for io module

"""

from tracemalloc import take_snapshot
import numpy as np
import pytest
from pathlib import Path
from genericpath import isfile
import json as json

import timescaleanalysis
import timescaleanalysis.io

TEST_TRAJ = Path(__file__).parent / 'test_data/test_trajectories'
TEST_DATA = Path(__file__).parent / 'test_data'