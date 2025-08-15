# GB_kinetics
Implementation of disconnection image creation algorithm outlined in the paper:<br />
"Energetics of the nucleation and glide of disconnection modes in symmetric tilt grain boundaries". <br />
The paper in under review and a pre-print is available at http://dx.doi.org/10.13140/RG.2.2.23055.09129 <br />
<br />
<br />

Documentation for this project is hosted at : https://hjoshi9.github.io/GB_kinetics/ <br />

NOTE : This is the OOP implementaiton of GB kinetics code. The repository is under active development!!
<br />
<br />

The workflow for GB Kinetics can be broken down in 5 steps:

1. Construction of STGB microstate (run gridSearch.py)<br />
2. Enumeration of disconnection modes (implemented in oILAB, available on the Github repository: https://github.com/oiLAB-project/oILAB)<br />
3. Atomistic construction of a disconnection mode <br />
4. Mapping the atomic shuffles <br />
5. Evaluation of the energy barrier and trajectories using NEB <br />
GB_kinetics.py carries out the steps 2-4.

<br />
<br />
Necessary installations: <br />
The program requires python3, LAMMPS and openmpi installations on the system. <br />
Details on installation of LAMMPS can be found at https://docs.lammps.org/Install.html <br />
Details on installation of openmpi can be found at https://docs.open-mpi.org/en/v5.0.x/installing-open-mpi/quickstart.html<br />
Necessary packages for python code are listed in requirements.txt  and can be installed by using :  pip install -r requirements.txt<br />
<br />
