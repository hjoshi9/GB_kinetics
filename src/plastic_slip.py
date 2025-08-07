import numpy as np
import math

class dislocation_dipole:
    """
        Class to calculate plastic slip due to a dislocation dipole.

        Attributes:
            dislocation_points (np.ndarray): Array of coordinates for dislocations (2D points).
            period (float): Periodicity of the system (typically along GB direction).
            burgers_vector (float): Magnitude of the Burgers vector.
            dislocation_vector (np.ndarray): Vector from first to second dislocation.
            dislocation_distance (float): Distance between the two dislocations.
    """
    def __init__(self,dislocation_points, period, burgers_vector):
        """
            Initialize a dislocation dipole object.

            Args:
                dislocation_points (np.ndarray): Coordinates of the dislocation pair (shape: 2x2).
                period (float): Periodicity of the system (along grain boundary direction).
                burgers_vector (float): Magnitude of the Burgers vector.
        """
        self.dislocation_points = dislocation_points
        self.period = period
        self.burgers_vector = burgers_vector
        self.dislocation_vector = dislocation_points[0]-dislocation_points[1]
        self.dislocation_distance = np.linalg.norm(self.dislocation_vector)

    def _local_position(self,disloc1,disloc2,x):
        """
            Transform a point `x` into a local coordinate system based on a dislocation segment.

            Args:
                disloc1 (np.ndarray): Starting point of the dislocation segment.
                disloc2 (np.ndarray): Ending point of the dislocation segment.
                x (np.ndarray): Point to transform into local coordinates.

            Returns:
                tuple:
                    - np.ndarray: Local coordinates [parallel, perpendicular] relative to the segment.
                    - float: Half length of the dislocation segment.
        """
        A2B = self.dislocation_vector
        normA2B  = self.dislocation_distance
        t = A2B / normA2B
        n = np.array([-t[1], t[0]])
        c = 0.5 * (disloc1 + disloc2)
        return np.array([np.dot(x - c, t), np.dot(x - c, n)]), 0.5 * normA2B

    def plastic_displacement(self,x,nImages=2):
        """
            Compute the plastic displacement at a point `x` due to a dislocation dipole.

            Args:
                x (np.ndarray): 2D point at which displacement is calculated.
                nImages (int, optional): Number of periodic image dipoles to consider. Defaults to 2.

            Returns:
                tuple:
                    - float: Total angle of plastic distortion (radians).
                    - float: Plastic displacement at the point `x`.
        """
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
