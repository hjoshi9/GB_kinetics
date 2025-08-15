GB Kinetics Example
=============================

This example demonstrates how to run the GB kinetics script and walks through where the results are stored.

- Open GB_kinetics.py and input Gb information, displacements recorded earlier and parameters for optimal transport calculations. Start with specifing the parameters required to identify the material and GB such as sigma number, misorientation, inclination, material symbol, and tilt axis. ``size_along_gb_period`` determines the length of the box along the gb period. The length of the box generated is 2*period*size_along_gb_period. ``size_along_tilt_axis`` determines the length of the box along the tilt axis. The length of the box generated is 2*(CSL length along tilt axis)*size_along_tilt_axis. ``lattive_Vectors`` determine what crystal system is under consideration.

.. literalinclude:: ../../GB_kinetics.py
   :language: python
   :lines: 4-24
   :linenos:

- Next specify the parameters for Sinkhorn Algorithm run which determines the atomic trajectories in transformed region. ``regularizationParameter`` determines the regularization for Sinkhorn algorithm. It can be loosely thought of as a measure of temperature. Lower ``regularizationParameter`` means algorithm is more contrained and gives out only the smallest displacement maps. Higher ``regularizationParameter`` might give out multiple atomic mappings some of them might not be the lowest displacement. ``maximumIterations`` determines the maximum number of iterations done by Sinkhorn Algorithm. If the algorithm does not converge (code will raise this error), increase this parameter. ``chooseDisconnection`` is an optional argument which allows users to choose the disconnection mode if they want. Set ``chooseDisconnection = True`` if you want to choose the disconnection mode, ``choose_disconnection = False`` will automatically create disconnection mode with smallest burgers vector and corresponding step height.

.. literalinclude:: ../../GB_kinetics.py
   :language: python
   :lines: 26-33
   :linenos:

- Specify the file that contains bicrystallography data from oilab output in ``oilab_output_file``. Current implementation has output files included for 3 tilt axes ([001],[110],[111]), stored in ``Data`` folder. ``output_folder`` is the folder in which the output of this program is stored.

.. literalinclude:: ../../GB_kinetics.py
   :language: python
   :lines: 35-40
   :linenos:

- Specify locations of external programs needed for running this code. This includes location of lammps installation, mpi installation. Also indicate the full path to the lammps potential file you want to use.

.. literalinclude:: ../../GB_kinetics.py
   :language: python
   :lines: 42-48
   :linenos:

- Input the displacements that minimize GB to minimum energy structure. Can be obtained using ``gridSearch.py`` script.

.. literalinclude:: ../../GB_kinetics.py
   :language: python
   :lines: 50-52
   :linenos:

- Specify the parameters for NEB run. ``partitions`` is the linearly interpolated images used in NEB run. ``neb_mode`` lets you choose if you want to run intermediate images through NEB or not. ``neb_mode`` = 1 -> NEB with intermediate images, ``neb_mode`` = 0 -> NEb with just the initial and final GB images

.. literalinclude:: ../../GB_kinetics.py
   :language: python
   :lines: 54-61
   :linenos:

- Finally call the function that calls the required functions and carries out the procedure detailed in the paper.

.. literalinclude:: ../../GB_kinetics.py
   :language: python
   :lines: 63-83
   :linenos:

* Save and run GB_kinetics.py

.. code-block:: console

    $ python3 GB_kinetics.py

User input
-----------
After running ``GB_kinetics.py``, the program will show the input parameters pertinent to system
user wants to explore.

.. image:: /_static/user_input1.png
   :alt: Disconnection images without any elastic relaxation
   :width: 600px

Then the program asks the user to choose the disconnection mode they want to construct. To do this, it
lists out the disconnection modes calculated using GB information

.. image:: /_static/user_input2.png
   :alt: Disconnection images without any elastic relaxation
   :width: 600px

User can input the values of ``m`` and ``n``, that corresponds to the disconnection mode they want to
construct. The program confirms the disconnection mode entered.

.. image:: /_static/user_input3.png
   :alt: Disconnection images without any elastic relaxation
   :width: 600px

Command line Output
-------------------
The code prints out which configuration it is generating and when completed, prints out the
location of the output generated.

.. image:: /_static/user_input4.png
   :alt: Disconnection images without any elastic relaxation
   :width: 600px

Output
-------

This code generates atomic configurations of different types:

#. LAMMPS data files for bicrystal with atoms obtained from bicrystallographic data and an applied plastic displacement , as discussed in :doc:`theory`.

#. LAMMPS data files for minimized bicrystal with atoms now representing GB ground state.

#. A LAMMPS data file containing multiple timesteps showing minimization of the bicrystal generated from bicrystallography.

#. LAMMPS data files for bicrystals with atomic trajectories calculated using min-shuffle algorithm.

#. Data file corresponding to the bicrystals with atomic trajectories, containing the atoms that are to be included in ``neb`` calculations.

All the generated configurations are stored in:
``output/<Element>/Sigma<GB sigma number>/Misorientation<GB misorientation>.0/size<Size of box along GB>/b<Disconnection burgers vector>h<disconnection step height>``

- The first output is the disconnection images constructed using bicrytallographic information. No elastic fields are applied to these images.

.. image:: /_static/sigma17_ascut.gif
   :alt: Disconnection images without any elastic relaxation
   :width: 800px

- Then the code minimizes the images using displacements derived from gridSearch run

.. image:: /_static/sigma17_min.gif
   :alt: Disconnection images without any elastic relaxation
   :width: 800px

- The code then finds out the atomic trajectories using min-shuffle algorithm

.. image:: /_static/sigma17_min_shuffle.gif
   :alt: Disconnection images without any elastic relaxation
   :width: 800px

- Finally it uses the images generated to run a neb calculation on the images.

.. image:: /_static/sigma17_size1_nebresults.gif
   :alt: NEB GB migration mechanism
   :width: 600px

- The post processing function also plots the MEP w.r.t the reaction coordinates

.. image:: /_static/Cusigma17_mis28.0_size1_discb0.88h1.75_partition4.png
   :alt: Disconnection images without any elastic relaxation
   :width: 600px
