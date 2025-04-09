import numpy as np 
import gzip 
from pathlib import Path

def read_traj(t):
    Nskip = 9 
    
    Config = []
    Box = []
    frame_nr_old = -1 
    mfile = Path(t)
    if mfile.is_file():
        with open(t, "r") as traj_file:
            Natoms = 1000
            try: 
                for i,line in enumerate(traj_file):
                    modulo = i % (Nskip+Natoms)
                    frame_nr = i // (Nskip+Natoms)
                    if frame_nr != frame_nr_old:
                        Config.append([])
                    if modulo == 3:
                        Natoms = np.array(line.split()).astype(float)[0]

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
            if len(Config[-1])!=Natoms:
                del Config[-1]
                del Box[-1]

    Config = np.array(Config)
    return Natoms, Config, Box 

def read_bop(t,Natoms):
    Nskip = 9 
    BOP = []
    frame_nr_old = -1 
    mfile = Path(t)
    if mfile.is_file():
        with open(t, "r") as traj_file:
     
            try: 
                for i,line in enumerate(traj_file):
                    modulo = i % (Nskip+Natoms)
                    frame_nr = i // (Nskip+Natoms)
                    if frame_nr != frame_nr_old:
                        BOP.append([])
                    
                    if modulo >=Nskip:
                        whole_line = np.array(line.split()).astype(float)
                        E_perpart = whole_line[1]
                        q1 = whole_line[2]
                        q2 = whole_line[3] 
                        q3 = whole_line[4]
                        q4 = whole_line[5]
                        q5 = whole_line[6]
                        q6 = whole_line[7] 
                        q7 = whole_line[8] 
                        q8 = whole_line[9]
                        BOP[-1].append(np.array([E_perpart,q1,q2,q3,q4,q5,q6,q7,q8])) 

                    frame_nr_old = frame_nr
            except EOFError as er:
                print(er)
        
        if BOP:
            if len(BOP[-1])!=Natoms:
                del BOP[-1]
             
    BOP = np.array(BOP)
    return BOP
