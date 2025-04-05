import numpy as np
import math

def local_position(disloc1,disloc2,x):
    """
    Calculate the local position of an atom with respect to the dislocation dipole
    """
    A2B = -(disloc2 - disloc1)
    normA2B = np.linalg.norm(A2B)
    t = A2B/normA2B
    n = np.array([-t[1],t[0]])
    c = 0.5*(disloc1+disloc2)
    return np.array([np.dot(x-c,t),np.dot(x-c,n)]),0.5*normA2B

def solidangle_displacement(nImages,disloc_points,period,x,b):
    """
    Calculates the solid angle subtended by a point onto the dislocation dipole and thus finds the plastic displacement of the particle
    """
    angle = 0
    disloc1 = disloc_points[0]
    disloc2 = disloc_points[1]
    A2B = -(disloc2 - disloc1)
    normA2B = np.linalg.norm(A2B)
    if abs(normA2B)<1e-6:
        displacement = 0
    else:
        for i in range(-nImages,nImages+1,1):
            for k in range(len(disloc_points)-1):
                disloc1 = disloc_points[k]
                disloc2 = disloc_points[k+1]
                xL,halfLength = local_position(disloc1, disloc2, x+i*period*100)
                if abs(xL[1])<1e-16:
                    if abs(xL[0])<halfLength:
                        if xL[1]>0:
                            angle += 2.0*np.pi
                        elif xL[1]<0:
                            angle -= 2.0*np.pi
                        else:
                            angle += 0.0
                else:
                    Yterm = abs(1.0/xL[1])
                    angle += 2*np.sign(xL[1])*(-math.atan((xL[0]-halfLength)*Yterm) + math.atan((xL[0]+halfLength)*Yterm))
        displacement = -(angle/(4.0*np.pi))*b
    return angle,displacement
