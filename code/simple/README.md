# Simple model simulation

Here are the notebooks, data, and results for the simple model application.
The results obtained in this application is in `results`.

## Data

The model and data contents are in `model.pickle` and `data.pickle`, respectively.
Additionally, `grid.pickle` contains the horizontal Cartesian coordinates of the 
observations.

## Notebooks

### Calculations:

* [model.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/simple/model.ipynb):
  Builds the true source.
* [data.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/simple/data.ipynb):
  Calculates the data produced by the true source.
* [multiple_inversion.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/simple/multiple_inversion.ipynb):
  Perform the inversion for a grid of tentative values of $m_0$ and $z_0$.
* [single_inversion.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/simple/single_inversion.ipynb):
  Perform the inversion for a pair of $m_0$ and $z_0$.
* [RTP_anomaly.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/simple/RTP_anomaly.ipynb):
  Calculates the RTP anomaly.


### Plots:

* [plot_model_data.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/simple/plot_model_data.ipynb):
  Plots a figure with the observed data, the true source, the map of $\Gamma$,
and the RTP anomaly.
* [plot_solutions.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/simple/plot_solutions.ipynb):
  Plots the results of the inversion.
* [plot_validation.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/simple/plot_validation.ipynb):
  Plots the grid of tentative values of $m_0$ and $z_0$.
* [single_vizualiation.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/simple/single_vizualiation.ipynb):
  Plots the results of the single inversion.

