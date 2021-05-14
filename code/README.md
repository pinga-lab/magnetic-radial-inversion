# Source code for producing the results and figures

The source codes listed below contain the inversion algorithm which is the 
aim of the paper and some suplementary functions.

## Codes

* `mag_polyprism_functions.py` - contains several functions to build models, 
	generate synthetic data, and run the inversion algorithm.
* `mag_spheres_functions.py` - contains functions to perform the equivalent layer 
	processing technique.
* `plot_functions.py` - contains some functions to plot the results.

* `test_mag_polyprism_functions.py` - tests of the functions using pytest.


## Folders

All foders here contain the notebooks, data, and results of each application
presented in the manuscript.

* `anitapolis` - field application in Southern Brazil.
* `complex` - synthetic application of a source with a complex geometry.
* `dipping` - synthetic application of a source with a low dipping.
* `dipping-regional` - synthetic application of a source with a low dipping 
	in the presence of a regional data.
* `simple` - synthetic application of a source with a simple geometry.



## Notebooks

* [constraint_functions_example.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/constraint_functions_example.ipynb):
  Examples of the Hessian matrix of the constraints.

The `Makefile` has a rule to convert all notebooks to PDF.
To convert, run the following in this directory:

    make

