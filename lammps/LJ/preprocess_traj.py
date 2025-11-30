import keras
import numpy as np
import pandas as pd
from lammpstools import distances, neighbours, read_traj, pair_distances
from scipy.spatial.distance import squareform
import argparse 

parser = argparse.ArgumentParser 
parser.add_argument(-n, dtype=int)
parser.parse_args()

N_pick = int(parser.n)
Min_distance = 5 
Natoms, Config, Box = read_traj("trajectory.lammpstrj")
#N_steps = len(Config)
N_steps = 1 
all_dists = []
for ti in range(N_steps):
    # first pick particle at random
    id=np.random.randint(1,Natoms+1)
    # initiate list of picked particles
    picked = [id]
    while len(picked)<N_pick:
        #pick next particle at random 
        id_candidate = np.random.randint(1,Natoms+1)
        dist_ij = 1000 
        if id_candidate not in picked:
            for id_check in picked:
                dist_ij = pair_distances(id_check, id_candidate,Config[ti]*Box[ti], Box[ti])
                if dist_ij < Min_distance:
                    break
            # check if all picked ids were checked and if yes:
            # add candidate to picked ids 
            if id_check == picked[-1]:
                picked.append(id_candidate)
    
print(picked)
                
            




