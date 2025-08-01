import numpy as np
import math

class dislocation_dipole:
    """
    Class to calculate plastic slip due to a dislocation dipole
    """
    def __init__(self,dislocation_points, period, burgers_vector):
        self.dislocation_points = dislocation_points
        self.period = period
        self.burgers_vector = burgers_vector
        self.dislocation_vector = dislocation_points[0]-dislocation_points[1]
        self.dislocation_distance = np.linalg.norm(self.dislocation_vector)

    def _local_position(self,disloc1,disloc2,x):
        A2B = self.dislocation_vector
        normA2B  = self.dislocation_distance
        t = A2B / normA2B
        n = np.array([-t[1], t[0]])
        c = 0.5 * (disloc1 + disloc2)
        return np.array([np.dot(x - c, t), np.dot(x - c, n)]), 0.5 * normA2B

    def plastic_displacement(self,x,nImages=2):
        angle = 0
        disloc_points = self.dislocation_points
        b = self.burgers_vector
        period = self.period
        normA2B = self.dislocation_distance
        if abs(normA2B) < 1e-6:
            displacement = 0
        else:
            for i in range(-nImages, nImages + 1, 1):
                for k in range(len(disloc_points) - 1):
                    disloc1 = disloc_points[k]
                    disloc2 = disloc_points[k + 1]
                    xL, halfLength = self._local_position(disloc1,disloc2,x + i * period*100)
                    if abs(xL[1]) < 1e-16:
                        if abs(xL[0]) < halfLength:
                            if xL[1] > 0:
                                angle += 2.0 * np.pi
                            elif xL[1] < 0:
                                angle -= 2.0 * np.pi
                            else:
                                angle += 0.0
                    else:
                        Yterm = abs(1.0 / xL[1])
                        angle += 2 * np.sign(xL[1]) * (-math.atan((xL[0] - halfLength) * Yterm) +
                                                       math.atan((xL[0] + halfLength) * Yterm))
            displacement = -(angle / (4.0 * np.pi)) * b
        return angle, displacement
