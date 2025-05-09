import numpy as np 
import gzip 
from pathlib import Path
import numba 
from scipy.spatial.distance import squareform


def read_traj(t):
    Nskip = 9 
    
    Config = []
    Box = []
    frame_nr_old = -1 
    mfile = Path(t)
    Natoms = 0 
    if mfile.is_file():
        with open(t, "r") as traj_file:
            try: 
                for i,line in enumerate(traj_file):
                    modulo = i % (Nskip+Natoms)
                    frame_nr = i // (Nskip+Natoms)
                    if frame_nr != frame_nr_old:
                        Config.append([])
                        Box.append([])
                        
                    if modulo == 3:
                        Natoms = np.array(line.split()).astype(float)[0]

                    if modulo == 5:
                        whole_line = np.array(line.split()).astype(float)
                        Lstart = whole_line[0]
                        Lend = whole_line[1]
                        Lx = Lend - Lstart
                        Box[-1].extend([Lx])
                        
                    if modulo == 6:
                        whole_line = np.array(line.split()).astype(float)
                        Lstart = whole_line[0]
                        Lend = whole_line[1]
                        Ly = Lend - Lstart
                        Box[-1].extend([Ly])
                        
                    if modulo == 7:
                        whole_line = np.array(line.split()).astype(float)
                        Lstart = whole_line[0]
                        Lend = whole_line[1]
                        Lz = Lend - Lstart
                        Box[-1].extend([Lz])
                    

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
                        BOP[-1].append(np.array(whole_line[1:]))

                    frame_nr_old = frame_nr
            except EOFError as er:
                print(er)
        
        if BOP:
            if len(BOP[-1])!=Natoms:
                del BOP[-1]
             
    BOP = np.array(BOP)
    return BOP


@numba.njit(fastmath=True, parallel=False)
def distances(frame_i, Box):
    
    lx_box = Box[0]
    ly_box = Box[1]
    lz_box = Box[2]
    
    dist_norm = []    
    for i, ipos in enumerate(frame_i):
        for j, jpos in enumerate(frame_i):
            if j>i: 
                dist = ipos - jpos
                
                dx = dist[0]
                dy = dist[1]
                dz = dist[2]
                
                sign_dx = np.sign(dx)
                sign_dy = np.sign(dy)
                sign_dy = np.sign(dz)
                
                # pbc only for x and y 
                dx = sign_dx*(min(np.fabs(dx),lx_box-np.fabs(dx)))
                dy = sign_dy*(min(np.fabs(dy),ly_box-np.fabs(dy)))
                dz = sign_dy*(min(np.fabs(dz),lz_box-np.fabs(dz)))

                dist_ij = np.sqrt(dx*dx+dy*dy+dz*dz)
                dist_norm.append(dist_ij)
                
    return dist_norm
    
    
@numba.njit(fastmath=True, parallel=False)
def vector_squareform_distances(frame_i, Box):
    
    lx_box = Box[0]
    ly_box = Box[1]
    lz_box = Box[2]
    
    dist_norm = [] 
    vdist = []   
    for i, ipos in enumerate(frame_i):
        for j, jpos in enumerate(frame_i):
                dist = ipos - jpos
                
                dx = dist[0]
                dy = dist[1]
                dz = dist[2]
                
                sign_dx = np.sign(dx)
                sign_dy = np.sign(dy)
                sign_dy = np.sign(dz)
                
                # pbc only for x and y 
                dx = sign_dx*(min(np.fabs(dx),lx_box-np.fabs(dx)))
                dy = sign_dy*(min(np.fabs(dy),ly_box-np.fabs(dy)))
                dz = sign_dy*(min(np.fabs(dz),lz_box-np.fabs(dz)))

                dist_ij = np.sqrt(dx*dx+dy*dy+dz*dz)
                dist_norm.append(dist_ij)    
                vdist.append([dx,dy,dz])
                          
    return dist_norm, vdist
      

def neighbours(sq_dist, cutoff):
    b = np.where((sq_dist<cutoff) & (sq_dist>0.01))
    neighbour_list = [[b[0][i],b[1][i]] for i in range(len(b[0]))]
    return neighbour_list

def nextN_neighbours(Natoms, sq_dist,nn):
    NextN = np.zeros((int(Natoms),int(nn)))
    for i in range(int(Natoms)):
        NextN[i] = np.sort(sq_dist[i])[1:(nn+1)]

    return NextN
    
def nextN_neighbours_vector(Natoms, sq_dist, vec_dist, nn):
    NextN = np.zeros((int(Natoms),int(nn),3))
    for i in range(int(Natoms)):
        idx = np.argsort(sq_dist[i])[1:(nn+1)]
        NextN[i] = vec_dist[i][idx]

    return NextN
        
