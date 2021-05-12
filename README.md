# Magnetic radial inversion for 3-D source geometry estimation

by
[Leonardo B. Vital](https://www.pinga-lab.org/people/vital.html)<sup>1</sup>,
[Vanderlei C. Oliveira Jr.](http://www.pinga-lab.org/people/oliveira-jr.html)<sup>1</sup>, and
[Valéria C. F. Barbosa](https://www.pinga-lab.org/people/barbosa.html)<sup>1</sup>

<sup>1</sup>[Observatório Nacional](http://www.on.br/index.php/pt-br/)

This repository contains the manuscript and supplementary code and data for the article "Magnetic radial inversion for 3-D source geometry estimation" submitted for publication in *Geophysical Journal International*.

> Brief description of what this paper is about (2-3 sentences). Include a
> figure as well with the main result of your paper.

![](complex.gif)

**Figure 1:** *Complex model simulation. The blue prisms represent the true model and the red prisms represent the estimated model.*


## Abstract

We present a method for inverting total-field anomaly data to estimate the geometry of 
a uniformly magnetized 3-D geological source in the subsurface. The method assumes 
the total-magnetization direction is known. 
We approximate the source by an ensemble of vertically juxtaposed right prisms, all of them with the same total-magnetization vector and depth extent. 
The horizontal cross-section of each prism is a polygon defined by a given number of
equi-angularly spaced vertices from $0^{\circ}$ to $360^{\circ}$,  whose polygon vertices 
are described by polar coordinates with an origin defined by a horizontal location 
over the top of each prism. 
Because our method estimates the radii of each polygon vertex  we refer to it as 
\textit{radial inversion}.
The position of these vertices, the horizontal location of each prism, and the depth extent of all prisms are the parameters to be estimated by solving a constrained nonlinear inverse problem of minimizing a goal function. 
We run successive inversions for a range of tentative total-magnetization intensities 
and depths to the top of the shallowest prism. The estimated models producing 
the lowest values of the goal function form the set of candidate solutions.
To obtain stabilized solutions, we impose the zeroth- and first-order Tikhonov 
regularizations on the shape of the prisms. The method allows estimating the geometry 
of both vertical and inclined sources, with a constant direction of magnetization, 
by using the Tikhonov regularization. 
Tests with synthetic data show that the method can be of utility in estimating the shape of the magnetic source even in the presence of a strong regional field.
Results obtained by inverting airborne total-field anomaly data over the 
Anit{\'a}polis alkaline-carbonatitic complex, in southern Brazil, 
suggest that the emplacement of the magnetic sources was controlled by NW-SE-trending 
faults at depth, in accordance with known structural features at the study area.


## Software implementation

This code runs a non-linear inversion algorithm and its suplementary functions.

All source code used to generate the results and figures in the paper are in
the `code` folder.
The folder`code` contains the folders `anitapolis`, `complex`, `dipping`, `dipping-regional`, and `simple`, which correspond to the paper applications.
The calculations and figure generation are all run inside
[Jupyter notebooks](http://jupyter.org/).
The data used in this study is provided in `data` and the sources for the
manuscript text and figures are in `manuscript`.
Results generated by the code are saved in `results` inside each application folder.
See the `README.md` files in each directory for a full description.


## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/pinga-lab/magnetic-radial-inversion.git

or [download a zip archive](https://github.com/pinga-lab/magnetic-radial-inversion/archive/master.zip).

A copy of the repository is also archived at *insert DOI here*


## Dependencies

You'll need a working Python environment to run the code.
The recommended way to set up your environment is through the
[Anaconda Python distribution](https://www.anaconda.com/download/) which
provides the `conda` package manager.
Anaconda can be installed in your user directory and does not interfere with
the system Python installation.
The required dependencies are specified in the file `environment.yml`.

We use `conda` virtual environments to manage the project dependencies in
isolation.
Thus, you can install our dependencies without causing conflicts with your
setup (even with different Python versions).

Run the following command in the repository folder (where `environment.yml`
is located) to create a separate environment and install all required
dependencies in it:

    conda env create


## Reproducing the results

Before running any code you must activate the conda environment:

    source activate radial-mag

or, if you're on Windows:

    activate radial-mag

This will enable the environment for your current terminal session.
Any subsequent commands will use software that is installed in the environment.

To build and test the software, produce all results and figures, and compile
the manuscript PDF, run this in the top level of the repository:

    make all

If all goes well, the manuscript PDF will be placed in `manuscript`.

The way to explore the code results is to execute the Jupyter notebooks
individually.
To do this, you must first start the notebook server by going into the
repository top level and running:

    jupyter notebook

This will start the server and open your default web browser to the Jupyter
interface. In the page, go into the `code` folder, select the desired application
folder, and select the
notebook that you wish to view/run.

The notebook is divided into cells (some have text while other have code).
Each cell can be executed using `Shift + Enter`.
Executing text cells does nothing and executing code cells runs the code
and produces it's output.
To execute the whole notebook, run all cells in order.


## License

All source code is made available under a BSD 3-clause license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

The manuscript text is not open source. The authors reserve the rights to the
article content, which is currently submitted for publication in the
JOURNAL NAME.
