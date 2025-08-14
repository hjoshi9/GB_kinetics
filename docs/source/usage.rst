Installation
============

Installations required
----------------------

| Repository is avaialble at https://github.com/hjoshi9/GB_kinetics.git
| Clone the repository

.. code-block:: console

	$ git clone -b main https://github.com/hjoshi9/GB_kinetics.git

| The program requires python3, LAMMPS and openmpi installations on the system.
| Details on installation of LAMMPS can be found at https://docs.lammps.org/Install.html
| Details on installation of openmpi can be found at https://docs.open-mpi.org/en/v5.0.x/installing-open-mpi/quickstart.html
| Necessary packages for python code are listed in requirements.txt and can be installed by 

.. code-block:: console

    $ pip install -r requirements.txt


Workflow
---------

| The method can be divided into 5 steps:

#. Construction of STGB microstate 
#. Enumeration of disconnection modes
#. Atomistic construction of a disconnection mode
#. Mapping the atomic shuffles
#. Evaluation of the energy barrier and trajectories using NEB.

These steps are discussed in detail in :doc:`Theory <theory>`

Usage
---------

#. Running gridSearch.py script to determine the displacements which need to be applied to as but bicrystal to get to lowest energy microstate.
#. Inputting the displacements in GB_kinetics.py script and running it to generate disconnection images and neb.

An example run of Grid search procedure is shown in :doc:`Grid Search Example <example_gs>`
and an example run of disconnection image generation is shown in :doc:`GB Kinetics Example<example_gbkinetics>`
