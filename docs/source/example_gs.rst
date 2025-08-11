Example: Running Grid Search
============================

This example demonstrates how to run grid search script and interpret the results.

* Open grid_search.py and input the Gb information and grid search parameters

Here is the source code of the example script:

- Open GB_kinetics.py and input Gb information, displacements recorded earlier and parameters for optimal transport calculations. Start with specifing the parameters required to identify the material and GB such as sigma number, misorientation, inclination, material symbol, and tilt axis. ``size_along_gb_period`` determines the length of the box along the gb period. The length of the box generated is 2*period*size_along_gb_period. ``size_along_tilt_axis`` determines the length of the box along the tilt axis. The length of the box generated is 2*(CSL length along tilt axis)*size_along_tilt_axis. ``lattive_Vectors`` determine what crystal system is under consideration.

.. literalinclude:: ../../gridSearch.py
   :language: python
   :lines: 6-27
   :linenos:

- Specify the file that contains bicrystallography data from oilab output in ``oilab_output_file``. Current implementation has output files included for 3 tilt axes ([001],[110],[111]), stored in ``Data`` folder. ``output_folder`` is the folder in which the output of this program is stored.

.. literalinclude:: ../../gridSearch.py
   :language: python
   :lines: 28-34
   :linenos:

- Specify locations of external programs needed for running this code. This includes location of lammps installation, mpi installation. Also indicate the full path to the lammps potential file you want to use.

.. literalinclude:: ../../gridSearch.py
   :language: python
   :lines: 35-42
   :linenos:

- Specify the parameters that control the scope of gridsearch and number of cores to be used. ``num_cores`` is the number of cores to be used when calling lammps. ``step_increments`` determines the resolution of the grid used in this grid search. ``limit`` is the highest extend upto which the code explores.

.. literalinclude:: ../../gridSearch.py
   :language: python
   :lines: 43-57
   :linenos:

- Finally call the function that calls the required functions.

.. literalinclude:: ../../gridSearch.py
   :language: python
   :lines: 58-78
   :linenos:

* Save and run grid_search.py

.. code-block:: console

    $ python3 grid_search.py

Output
-------

The script will generate several outputs, including:

- Atomic structure snapshots

.. image:: /_static/gridSearch_configs.png
   :alt: Minimized GB structures
   :width: 600px

- Contour plot of Energy landspace with displacements

.. image:: /_static/gridsearch.png
   :alt: Energy landscape obtained from Grid Search
   :width: 600px

* Record the displacement that gets the system to lowest energy state. This serves as an input to ``GB_kinetics.py`` script.
