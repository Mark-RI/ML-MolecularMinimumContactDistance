import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import random as rand
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

angmax=0.4
anglemu=1.570796327
nc=4
densigas=0.00025
densiliq=0.112
xl=1.338
rmaxgas=1.3
rmaxliq=1.3
rf=1000000000
xmu2=0
qq2=3
temp=0.45
volmax=1
nvol=1
ninser=10000

def get_vector(a, b):
    return [b[0] - a[0], b[1] - a[1], b[2] - a[2]]

def get_scalar_product(a, b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

def get_cross_product(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

def xnorm(a):
    ad = (a[0]*a[0]+a[1]*a[1]+a[2]*a[2])**0.5
    return [a[0]/ad, a[1]/ad, a[2]/ad]

def vp(a,b):
    c=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
    return c

def initia():
    global rmaxgas, rmaxliq, vol
    ngas, nliq, n, rxgas, rygas, rzgas, rxliq, ryliq, rzliq, ex, ey, ez, emx, emy, emz = fcc()
    volgas=ngas/densigas
    volliq=nliq/densiliq
    vol=volgas+volliq
    boxgas=volgas**(1./3.)
    boxliq=volliq**(1./3.)
    if boxliq>8:	
        cutliq=4
        cutgas=4
    else:
        cutliq=boxliq/2
        cutgas=boxliq/2 
    cutgas2=cutgas**2
    cutliq2=cutliq**2
    rmaxgas=rmaxgas/boxgas
    rmaxliq=rmaxliq/boxliq
    return ngas, nliq, n, rxgas, rygas, rzgas, rxliq, ryliq, rzliq, ex, ey, ez, emx, emy, emz, boxliq, boxgas, cutgas2, cutliq2, vol

def fcc():
    rroot3=1/np.sqrt(3)
    ngas=4*nc**3
    nliq=4*nc**3
    n = nliq+ngas

    cell  = 1.0 / nc
    cell2 = 0.5 * cell

    rx = [0,0,0,0]
    ry = [0,0,0,0]
    rz = [0,0,0,0]

    ex = [0,0,0,0]
    ey = [0,0,0,0]
    ez = [0,0,0,0]

    emx = [0,0,0,0]
    emy = [0,0,0,0]
    emz = [0,0,0,0]

    #Sublattice A
    rx[0] =  0.0
    ry[0] =  0.0
    rz[0] =  0.0
    ex[0] =  rroot3
    ey[0] =  rroot3
    ez[0] =  rroot3
    if anglemu != 0:
        emx[0] = 0.0
        emy[0] = ez[0]
        emz[0] = -ey[0]
        em=xnorm([emx[0], emy[0], emz[0]])
        emx[0]=em[0]
        emy[0]=em[1]
        emz[0]=em[2]
    else:
        emx[0] = ex[0]
        emy[0] = ez[0]
        emz[0] = ey[0]

    #Sublattice B
    rx[1] =  cell2
    ry[1] =  cell2
    rz[1] =  0.0
    ex[1] =  rroot3
    ey[1] =  -rroot3
    ez[1] =  -rroot3
    if anglemu != 0:
        emx[1] = ez[1]
        emy[1] = 0.0
        emz[1] = -ex[1]
        em=xnorm([emx[1], emy[1], emz[1]])
        emx[1]=em[0]
        emy[1]=em[1]
        emz[1]=em[2]
    else:
        emx[1] = ex[1]
        emy[1] = ez[1]
        emz[1] = ey[1]

    #Sublattice C
    rx[2] =  0.0
    ry[2] =  cell2
    rz[2] =  cell2
    ex[2] =  -rroot3
    ey[2] =  rroot3
    ez[2] =  -rroot3
    if anglemu != 0:
        emx[2] = ey[2]
        emy[2] = -ex[2]
        emz[2] = 0.0
        em=xnorm([emx[2], emy[2], emz[2]])
        emx[2]=em[0]
        emy[2]=em[1]
        emz[2]=em[2]
    else:
        emx[2] = ex[2]
        emy[2] = ez[2]
        emz[2] = ey[2]

    #Sublattice D
    rx[3] =  cell2
    ry[3] =  0.0
    rz[3] =  cell2
    ex[3] =  -rroot3
    ey[3] =  -rroot3
    ez[3] =  rroot3
    if anglemu != 0:
        emx[3] = 0.0
        emy[3] = -ez[3]
        emz[3] = ey[3]
        em=xnorm([emx[3], emy[3], emz[3]])
        emx[3]=em[0]
        emy[3]=em[1]
        emz[3]=em[2]
    else:
        emx[3] = ex[3]
        emy[3] = ez[3]
        emz[3] = ey[3]

    for iz in range(nc):
        for iy in range(nc):
            for ix in range(nc):
                for iref in range(4):
                    rx.append(rx[iref] + cell * (ix))
                    ry.append(ry[iref] + cell * (iy))
                    rz.append(rz[iref] + cell * (iz))
                    ex.append(ex[iref])
                    ey.append(ey[iref])
                    ez.append(ez[iref])
                    emx.append(emx[iref])
                    emy.append(emy[iref])
                    emz.append(emz[iref])
    del rx[:4]
    del ry[:4]
    del rz[:4]
    del ex[:4]
    del ey[:4]
    del ez[:4]
    del emx[:4]
    del emy[:4]
    del emz[:4]

    for i in range(ngas):
        rx[i] = (rx[i]-0.5+(cell2*.5))
        ry[i] = (ry[i]-0.5+(cell2*.5))
        rz[i] = (rz[i]-0.5+(cell2*.5))

    rxgas = rx.copy()
    rygas = ry.copy()
    rzgas = rz.copy()
    rxliq = rx.copy()
    ryliq = ry.copy()
    rzliq = rz.copy()

    return ngas, nliq, n, rxgas, rygas, rzgas, rxliq, ryliq, rzliq, ex, ey, ez, emx, emy, emz

def plotparticles(rx, ry, rz, rxc, ryc, rzc, title, boxgas=1,):
    figgas = plt.figure()
    ax = figgas.add_subplot(111, projection='3d')

    vectors = {}
    for i in range(len(rx)):
        vectors.update({i: {'point':[rx[i], ry[i], rz[i]], 'color': 'g'}})

    for i in range(len(rxc)):        
        vectors.update({i+.5: {'point':[rxc[i], ryc[i], rzc[i]], 'color': 'r'}})

    for label, props in vectors.items():
        ax.scatter(*props['point'], color=props['color'], alpha=1)

    ax.set_xlim([-.5*boxgas, .5*boxgas])
    ax.set_ylim([-.5*boxgas, .5*boxgas])
    ax.set_zlim([-.5*boxgas, .5*boxgas])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid()
    ax.set_title(title)

def minimum_contact_distance(s, zn, zm):
    d=xl
    zr=d/2
    xx=s.clone()
    wn=torch.sum(zn*zm, dim=0)
    auxi=abs(wn)

    w = torch.cross(zn, zm, dim=0)
    w=w/torch.norm(w, dim=0, keepdim=True)
    z = torch.cross(zn, w, dim=0)
    z=z/torch.norm(z, dim=0, keepdim=True)
    y = torch.cross(w, zm, dim=0)
    y=y/torch.norm(y, dim=0, keepdim=True)
    a=torch.sum(z*y, dim=0)*-1
    b=torch.sum(s*z, dim=0)
    c=torch.sum(s*y, dim=0)
    tt=(a*b+c)/(a*a-1)
    t=(a*c+b)/(1-a*a)
    z=torch.where(t<0, z*-1, z)
    t=torch.where(t<0, t*-1, t)
    y=torch.where(t<0, y*-1, y)
    tt=torch.where(tt<0, tt*-1, tt)
    p=s+tt*y-t*z
    pq=torch.sum(p*p, dim=0)
    v=zr*zr-t*t
    vv=zr*zr-tt*tt
    aaa=torch.sum(s*zm, dim=0)
    bbb=torch.sum(s*zn, dim=0)
    aa=-zr*torch.sum(z*zm, dim=0)+aaa
    bb=zr*torch.sum(y*zn, dim=0)+bbb
    cc=torch.sum(s*s, dim=0)
    v1q=cc+zr*zr-aa*aa-2*zr*torch.sum(s*z, dim=0)
    v2q=cc+zr*zr-bb*bb+2*zr*torch.sum(s*y, dim=0)

    a=zr
    b=zr
    o=pq**0.5
    u=torch.sum(z*y, dim=0)

    def iterator():
        a=zr
        b=zr
        for _ in range(10):
            c=(tt-(t-a)*u)/(o-(zr*zr-a*a)**0.5)
            d=(c*c*zr*zr/(1+c*c))**0.5
            c=(t-(tt-d)*u)/(o-(zr*zr-d*d)**0.5)
            e=(c*c*zr*zr/(1+c*c))**0.5
            a=e
            b=d
        sa=(zr*zr-a*a)**0.5
        sb=(zr*zr-b*b)**0.5
        ss=pq+2*zr*zr-a*a-b*b+(t-a)*(t-a)+(tt-b)*(tt-b)-2*(t-a)*(tt-b)*u-2*o*(sa+sb)+2*sa*sb
        return ss
    
    def iterator2():
        a=zr
        b=zr
        sa=(zr*zr-a*a)**0.5
        sb=(zr*zr-b*b)**0.5
        ss=pq+2*zr*zr-a*a-b*b+(t-a)*(t-a)+(tt-b)*(tt-b)-2*(t-a)*(tt-b)*u-2*o*(sa+sb)+2*sa*sb
        return ss
    
    def iterator3():
        a=zr
        b=zr
        for _ in range(10):
            c=(tt-(t+a)*u)/(o-(zr*zr-a*a)**0.5)
            d=(c*c*zr*zr/(1+c*c))**0.5
            c=-(t-(tt-d)*u)/(o-(zr*zr-d*d)**0.5)
            e=(c*c*zr*zr/(1+c*c))**0.5
            a=e
            b=d
        sa=(zr*zr-a*a)**0.5
        sb=(zr*zr-b*b)**0.5
        rr=(t+a)*(t+a)+(tt-b)*(tt-b)-2*(t+a)*(tt-b)*u+(o-sa-sb)*(o-sa-sb)
        return rr
    
    def iterator4():
        a=zr
        b=zr
        sa=(zr*zr-a*a)**0.5
        sb=(zr*zr-b*b)**0.5
        rr=(t+a)*(t+a)+(tt-b)*(tt-b)-2*(t+a)*(tt-b)*u+(o-sa-sb)*(o-sa-sb)
        return rr
    
    def iterator5():
        a=zr
        b=zr
        for _ in range(10):
            c=-(tt-(t-a)*u)/(o-(zr*zr-a*a)**0.5)
            d=(c*c*zr*zr/(1+c*c))**0.5
            c=(t-(tt+d)*u)/(o-(zr*zr-d*d)**0.5)
            e=(c*c*zr*zr/(1+c*c))**0.5
            a=e
            b=d
        sa=(zr*zr-a*a)**0.5
        sb=(zr*zr-b*b)**0.5
        rr=(t-a)*(t-a)+(tt+b)*(tt+b)-2*(t-a)*(tt+b)*u+(o-sa-sb)*(o-sa-sb)
        return rr
    
    def iterator6():
        a=zr
        b=zr
        sa=(zr*zr-a*a)**0.5
        sb=(zr*zr-b*b)**0.5
        rr=(t-a)*(t-a)+(tt+b)*(tt+b)-2*(t-a)*(tt+b)*u+(o-sa-sb)*(o-sa-sb)
        return rr
    
    ss_o=torch.where(o!=0, iterator(), iterator2())

    ss_tt=torch.where((tt>t), torch.where((o!=0), iterator3(), iterator4()), torch.tensor(1e10))

    ss_t=torch.where((t>=tt), torch.where((o!=0), iterator5(), iterator6()), torch.tensor(1e10))

    stacked = torch.stack([ss_o, ss_tt, ss_t], dim=0)
    min_tensor = torch.min(stacked, dim=0).values

    min_t=((min_tensor)**0.5).clone()

    min_no_nan = torch.nan_to_num(min_t, nan=1e10)

    ss_3=(min_tensor)**0.5

    ss3=torch.where((tt>zr) & (v2q<zr**2), abs(bb), ss_3)
    ss2=torch.where((t>zr) & (v1q<zr**2), abs(aa), ss3)
    ss_2=torch.where((v>0) & (vv>0) & (v+vv+2*(v*vv)**0.5>pq), 0, ss2)

    rv=torch.where(auxi>0.9999999, torch.sum(s*zn, dim=0), ss_2)
    xc=(zn*rv)
    sc=(xx-xc)
    sc=sc/torch.norm(sc, dim=0, keepdim=True)
    auxi = (torch.sum((xc-xx)*(xc-xx), dim=0))**0.5
    ss1=torch.where(auxi<d, rv, (torch.sum(((xx-(zr*sc))-((xc+(zr*sc))-rv*zn))*((xx-zr*sc)-((xc+zr*sc)-rv*zn)), dim=0))**0.5)
    ss_1=torch.where(rv==0, ((torch.sum(s*s, dim=0))**0.5)-d, ss1)

    stacked_final = torch.stack([min_no_nan, ss_1], dim=0)
    min_contact_distance_tensor = torch.min(stacked_final, dim=0).values
    return(abs(min_contact_distance_tensor))

def energy(coordinate_matrix, orientation_matrix, box, n):
    i, j = torch.triu_indices(n, n, offset=1)
    coord_i = coordinate_matrix[:, i]
    coord_j = coordinate_matrix[:, j]
    separation_matrix = coord_i - coord_j
    minimum_image_matrix=separation_matrix-1*torch.round(separation_matrix/1)
    scaled_matrix = minimum_image_matrix*box
    distance_matrix = (torch.sum(scaled_matrix**2, dim=0, keepdim=True))**0.5
    zn = orientation_matrix[:, i]
    zm = orientation_matrix[:, j]
#    minimum_contact_distance_matrix = 
    energy_matrix = ((1/distance_matrix)**12-(1/distance_matrix)**6)*4
    loss = torch.sum(energy_matrix)
    total_distance = torch.sum(distance_matrix)
    return loss, total_distance, distance_matrix, minimum_image_matrix, separation_matrix, energy_matrix, scaled_matrix, box

def add_disk(normal, point, radius, color, ax):
    theta = np.linspace(0, 2*np.pi, 50)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(theta)

    if np.allclose(normal, [0, 0, 1]):
        x_rot, y_rot, z_rot = x, y, z
    else:
        z_axis = np.array([0, 0, 1])
        rot_axis = np.cross(z_axis, normal)
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        rot_angle = np.arccos(np.dot(z_axis, normal))
        
        points = np.vstack([x, y, z])
        rot_matrix = Rotation.from_rotvec(rot_angle * rot_axis).as_matrix()
        rotated_points = rot_matrix @ points
        x_rot, y_rot, z_rot = rotated_points

    x_disk = x_rot + point[0]
    y_disk = y_rot + point[1]
    z_disk = z_rot + point[2]
    ax.plot3D(x_disk, y_disk, z_disk, linestyle='-', color=color, label='Circle')
