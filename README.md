# Disconnection_energetics
Implementation of disconnection image creation algorithm outlined in the paper:<br />
"Energetics of the nucleation and glide of disconnection modes in symmetric tilt grain boundaries". <br /> 
The paper in under review and a pre-print is available at http://dx.doi.org/10.13140/RG.2.2.23055.09129 <br />
The method (described in detail in the paper) can be broken down in 5 steps.<br />
Run grid_search.py to carry out Step 1.<br />
data folder contains output from Step 2 (implemented in oILAB, available on the Github repository: https://github.com/oiLAB-project/oILAB)<br />
Run main.py to carry out Step 3, 4 and 5.<br />
Output it stored in /output/<element>/Misorientation<misorientation>/size<size>/<disconnection_mode> folder <br />
<br />
Necessary installations: <br />
The program requires python3, LAMMPS and openmpi installations on the system. <br />
Details on installation of LAMMPS can be found at https://docs.lammps.org/Install.html <br />
Details on installation of openmpi can be found at https://docs.open-mpi.org/en/v5.0.x/installing-open-mpi/quickstart.html<br />
Necessary packages for python code are listed in requirements.txt  and can be installed by using :  pip install -r requirements.txt<br />
<br />
