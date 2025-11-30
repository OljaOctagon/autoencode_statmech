import keras
import numpy as np
import pandas as pd
from tools import distances, neighbours, read_traj, nextN_neighbours
from scipy.spatial.distance import squareform


Natoms, Config, Box = read_traj("trajectory.lammpstrj")
ti = 0
dist = distances(Config[ti] * Box[ti], Box[ti])
