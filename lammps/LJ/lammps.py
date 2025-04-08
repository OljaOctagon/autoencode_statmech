import numpy as np 
import gzip 
from pathlib import Path

def read_lammpstrj(t):
    Nskip = 9 
    
    Config = []
    Box = []
    frame_nr_old = -1 
    mfile = Path(t)
    if mfile.is_file():
        with open(t, "r") as traj_file:
            Natoms = float(traj_file[3])
            try: 
                for i,line in enumerate(traj_file):
                    modulo = i % (Nskip+Natoms)
                    frame_nr = i // (Nskip+Natoms)
                    if frame_nr != frame_nr_old:
                        Config.append([])

                    if modulo == 5:
                        whole_line = np.array(line.split()).astype(float)
                        Lstart = whole_line[0]
                        Lend = whole_line[1]
                        L = Lend - Lstart
                        Box.append(L) 


                    if modulo >=Nskip:
                        whole_line = np.array(line.split()).astype(float)
                        x = whole_line[2]
                        y = whole_line[3]
                        z = whole_line[4] 

                        Config[-1].append(np.array([x,y,z])) 

                    frame_nr_old = frame_nr
            except EOFError as er:
                print(er)
        
        if Config:
            if len(Config[-1])!=Natoms
                del Config[-1]
                del Box[-1]

    Config = np.array(Config)
    return Config 

