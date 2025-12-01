import keras
import numpy as np
import pandas as pd
from lammpstools.tools import distances, neighbours, read_traj, pair_distance
from scipy.spatial.distance import squareform
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('-n')
args = parser.parse_args()

N_pick = int(args.n)
Min_distance = 5 
Natoms, Config, Box = read_traj("trajectory.lammpstrj")
#N_steps = len(Config)
N_steps = 1 

def pick_particles(N_pick, Natoms, Ci, Bi):
    # first pick particle at random
    id=np.random.randint(1,Natoms+1)
    print("first id", id)
    # initiate list of picked particles
    picked = [id]

    while len(picked)<N_pick:
        #pick next particle at random 
        id_candidate = np.random.randint(1,Natoms+1)
        print("new candidate id ", id_candidate)
        dist_ij = 1000 
        if id_candidate not in picked:
            for id_check in picked:
                dist_ij = pair_distance(id_check, id_candidate,Ci,Bi)
                if dist_ij < Min_distance:
                    print("dist between {} and {}: {} : too small".format(id_candidate, id_check, dist_ij))
                    break
            # check if all picked ids were checked and if yes:
            # add candidate to picked ids 
            if id_check == picked[-1]:
                picked.append(id_candidate)
                print("picked so far: ", picked)
    
    return picked
#------------------------------------------------

nn=12
for istep in range(N_steps):
    # pick particles
    Ci = Config[istep]*Box[istep]
    Bi = Box[istep]
    picked = pick_particles(N_pick, Natoms, Ci, Bi)
    print(picked)

    for id_particle in picked:
        NextN = nextN_neighbours_per_id(id_particle, nn, Ci, Bi)
        print(NextN)
    # Calculate next neighbours of each particle 




