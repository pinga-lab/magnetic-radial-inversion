# Dipping model simulation

Here are the notebooks, data, and results for the dipping model application.
The results obtained in this application is in `results`.

## Data

The model and data contents are in `model.pickle` and `data.pickle`, respectively.


## Notebooks

### Calculations:

* [model.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/dipping/model.ipynb):
  Builds the true source.
* [data.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/dipping/data.ipynb):
  Calculates the data produced by the true source.
* [multiple_inversion.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/dipping/multiple_inversion.ipynb):
  Perform the inversion for a grid of tentative values of $m_0$ and $z_0$.
* [single_inversion.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/dipping/single_inversion.ipynb):
  Perform the inversion for a pair of $m_0$ and $z_0$.
* [RTP_anomaly.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/dipping/RTP_anomaly.ipynb):
  Calculates the RTP anomaly.


### Plots:

* [plot_model_data.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/dipping/plot_model_data.ipynb):
  Plots a figure with the observed data, vertical coordinates of the observations,
and the true source.
* [plot_rtp_map.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/dipping/plot_rtp_map.ipynb):
  Plots the RTP anomaly and $\Gamma$ map.
* [plot_solutions.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/dipping/plot_solutions.ipynb):
  Plots the results of the inversion.
* [plot_validation.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/dipping/plot_validation.ipynb):
  Plots the grid of tentative values of $m_0$ and $z_0$.
* [single_vizualiation.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/dipping/single_vizualiation.ipynb):
  Plots the results of the single inversion.

