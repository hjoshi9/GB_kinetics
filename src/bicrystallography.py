import numpy as np

class bicrystallography:
    """
        Class to calculate grain boundary (GB) properties based on bicrystallography.

        Attributes:
            sigma (int): Sigma value representing the coincidence site lattice (CSL) density.
            misorientation (float): Misorientation angle between grains (in degrees).
            inclination (float): Inclination angle of the grain boundary (in degrees).
            axis (np.ndarray): Rotation axis vector (3D).
            lattice_parameter (float): Lattice parameter of the crystal.
    """
    def __init__(self,sigma,misorientation,inclination,axis,lattice_parameter):
        """
            Initialize a bicrystallography object.

            Args:
                sigma (int): Sigma value representing the CSL density.
                misorientation (float): Misorientation angle (degrees).
                inclination (float): Inclination angle of the GB (degrees).
                axis (list or np.ndarray): Rotation axis vector.
                lattice_parameter (float): Lattice parameter of the crystal.
         """
        self.sigma = sigma
        self.misorientation = misorientation
        self.inclination = inclination
        self.axis = axis
        self.lattice_parameter = lattice_parameter

    @staticmethod
    def _read_data(path):
        """
            Read and parse grain boundary data from a text file (OILAB output format).

            Args:
                path (str): Path to the GB data file.

            Returns:
                list: A list of parsed rows, where each row contains GB properties.
        """
        i = 0
        j = -1
        data = []
        k = 0
        with open(path, 'r') as file:
            flag = 0

            for line in file:
                fields = line.split(' ')
                i = i + 1
                if (len(fields) == 3 and fields[0] == "Sigma"):
                    j = -1
                    flag = 1
                    row = []
                    sig = (float(fields[2]))

                if (fields[0] == "Misorientation="):
                    n = len(fields)
                    mis = (float(fields[n - 1]))

                if (len(fields) > 15):
                    nrow = []
                    if fields[2] != "Inclination" and fields[1] != "-------------":
                        flag = 2
                        for t in range(len(fields)):
                            if fields[t] != "":
                                j = j + 1
                                nrow.append(fields[t])

                if flag == 2:
                    row = [sig, mis, float(nrow[0]), float(nrow[7]), float(nrow[1]), float(nrow[2]),
                           float(nrow[3]),float(nrow[4]),float(nrow[5]), float(nrow[6]), float(nrow[8]),
                           float(nrow[9]), float(nrow[10]),float(nrow[14]), float(nrow[16]),float(nrow[17]),
                           float(nrow[18]), float(nrow[22]), float(nrow[23]),float(nrow[24]), float(nrow[25]),
                           float(nrow[29]), float(nrow[15]), float(nrow[11]),float(nrow[12]), float(nrow[13])]
                    data.append(row)
        return data

    def _find_ATGB_data(self,path,period_cutoff=50):
        """
            Filter GB data based on misorientation and period constraints.

            Args:
                path (str): Path to the OILAB output file.
                period_cutoff (float, optional): Maximum allowed CSL period. Defaults to 50.

            Returns:
                np.ndarray: Filtered and sorted GB data for acceptable ATGB configurations.
        """
        mis  = self.misorientation
        axis = self.axis

        # Read file
        data = np.asarray(self._read_data(path))

        # Append gbs based on period_cutoff
        atgb_data = []
        prev_inc = -10
        for j in range(len(data)):
            if abs(data[j, 1] - mis) < 0.2 and abs(data[j, 2] - prev_inc) > 3 and data[j, 3] < period_cutoff:
                atgb_data.append(
                    [data[j, 0], data[j, 1], data[j, 2], data[j, 3], data[j, 10], data[j, 11], data[j, 12], data[j, 13],
                     data[j, 22], data[j, 23], data[j, 24], data[j, 25]])
                prev_inc = data[j, 2]
            if abs(data[j, 1] - mis) < 0.2 and data[j, 2] == 45.0:
                atgb_data.append(
                    [data[j, 0], data[j, 1], data[j, 2], data[j, 3], data[j, 10], data[j, 11], data[j, 12], data[j, 13],
                     data[j, 22], data[j, 23], data[j, 24], data[j, 25]])
        max_inc_onfile = atgb_data[-1][2]

        # Extend the range from current to 90 degrees (for [001] and [111] gbs)
        for j in range(len(atgb_data)):
            if axis[0] == 0 and axis[1] == 0 and axis[2] == 1:
                if 45 + atgb_data[j][2] > max_inc_onfile and 45 + atgb_data[j][2] <= 90:
                    atgb_data.append(
                        [atgb_data[j][0], atgb_data[j][1], 45 + atgb_data[j][2], atgb_data[j][3], data[j][10],
                         data[j][11], data[j][12], data[j][13], data[j][22], data[j, 23], data[j, 24], data[j, 25]])
            if axis[0] == 1 and axis[1] == 1 and axis[2] == 1:
                if 60 + atgb_data[j][2] > max_inc_onfile and 60 + atgb_data[j][2] <= 90:
                    atgb_data.append(
                        [atgb_data[j][0], atgb_data[j][1], 60 + atgb_data[j][2], atgb_data[j][3], data[j][10],
                         data[j][11], data[j][12], data[j][13], data[j][22], data[j, 23], data[j, 24], data[j, 25]])

        # Sort the atgb data w.r.t inclination
        a = np.array(atgb_data)
        col = 2
        a = a[np.argsort(a[:, col])]

        return a

    def gb_props(self,oilab_output_file = "/data/fcc0-10.txt",choose_decision=1):
        """
            Calculate GB properties and optionally prompt user to choose a disconnection mode.

            Args:
                oilab_output_file (str, optional): Path to the OILAB output file. Defaults to "/data/fcc0-10.txt".
                choose_decision (bool, optional): If True, allows user to choose disconnection mode. Defaults to True.

            Returns:
                tuple:
                    - gb_data (np.ndarray): GB data including sigma, misorientation, inclination, etc.
                    - burgers_vector (float): Burgers vector magnitude of selected disconnection mode.
                    - step_height (float): Step height of the selected disconnection mode.
        """
        lat_par = self.lattice_parameter
        period_cutoff = 10
        atgb_data = self._find_ATGB_data(oilab_output_file,period_cutoff)
        gb_data = atgb_data[0, :]
        # GB properties
        p = atgb_data[0, 3] * lat_par
        b = atgb_data[0, 4] * lat_par
        h = atgb_data[0, 7] * lat_par
        H = atgb_data[0, 8] * lat_par
        print("+--------------------------------------------------------------------+")
        print("+                           GB Kinetics                              +")
        print("+ This program generates atomistic images for motion of GB mediated  +")
        print("+             by disconnection nucleation and glide                  +")
        print("+--------------------------------------------------------------------+")
        print("\n======================== GB information ==============================")
        print("Sigma = " + str(gb_data[0]))
        print("Misorientation = " + str(gb_data[1]))
        print("Inclination = " + str(gb_data[2]))
        print("Period = " + str(p))
        print("Smallest burgers vector (b) = " + str(b))
        print("Fundamental glide step height (hg) = " + str(h))
        print("Distance between CSL planes (H) = " + str(H))
        if choose_decision == False:
            m = 1
            n = 0
        else:
            print("\n================== Choose disconnection mode =========================")
            print("There exist infinitely many disconnection modes with the form:")
            print("               (b,h) = (m * b, m * hg + n * H)               ")
            print("where m and n are integers                  \n")
            print("Example of a few of the possible disconnection modes are:")
            headers = ["m", "n", "|b|", "h", "coupling factor"]
            table_contents = []
            # print("m \t n \t |b| \t h \t coupling factor")
            for i in range(-3, 3, 1):
                for j in range(-3, 3, 1):
                    if i == 0 and j == 0:
                        continue
                    br = i * b
                    sh = i * h + j * H
                    beta = br / sh
                    table_contents.append([i, j, np.round(br, 2), np.round(sh, 2), np.round(beta, 2)])
            for col in headers:
                print(col.ljust(10), end="")
            print()

            # Print table rows
            for i, row in enumerate(table_contents, start=1):
                for col in row:
                    print(str(col).ljust(10), end="")
                print()
            m = int(input("\nEnter the m corresponding to the disconnection mode:"))
            n = int(input("Enter the n corresponding to the disconnection mode:"))
        burgers_vector = m * b
        step_height = m * h + n * H
        print("Disconnection mode considered in this run :")
        print("(b,h) = (" + str(np.round(burgers_vector, 2)) + "," + str(np.round(step_height, 2)) + "), coupling factor = " + str(
            np.round(burgers_vector / step_height, 2)))
        return gb_data, burgers_vector, step_height