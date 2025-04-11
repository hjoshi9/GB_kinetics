Usage
=====

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


Running this script
-------------------

| The method can be divided into 5 steps

#. Construction of STGB microstate 
#. Enumeration of disconnection modes
#. Atomistic construction of a disconnection mode
#. Mapping the atomic shuffles
#. Evaluation of the energy barrier and trajectories using NEB.

So a typical workflow would look like

* Open grid_search.py and input the Gb information and grid search parameters 
* Save and run grid_search.py 

.. code-block:: console

    $ python3 grid_search.py

* Record the displacement that gets the system to lowest energy state.
* Open main.py and input Gb information, displacements recorded earlier and parameters for optimal transport calculations
* Save and run main.py

.. code-block:: console

    $ python3 main.py

* The results are stored in /output/ subdirectory.