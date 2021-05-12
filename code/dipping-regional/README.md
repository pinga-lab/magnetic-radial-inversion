# Dipping model in the presence of a regional field simulation

Here are the notebooks, data, and results for the dipping model in the 
presence of a regional field application.
The results obtained in this application is in `results`.

## Data

The data and regional-separation contents are in `data.pickle` and 
`dipping_regional_data.txt`, respectively.


## Notebooks

### Calculations:

* [data.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/dipping-regional/data.ipynb):
  Calculates the data produced by the true source.
* [multiple_inversion.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/dipping-regional/multiple_inversion.ipynb):
  Perform the inversion for a grid of tentative values of $m_0$ and $z_0$.
* [single_inversion.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/dipping-regional/single_inversion.ipynb):
  Perform the inversion for a pair of $m_0$ and $z_0$.
* [RTP_anomaly.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/dipping-regional/RTP_anomaly.ipynb):
  Calculates the RTP anomaly.
* [regional_separation.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/anitapolis/regional_separation.ipynb):
  Performs the regional separation and plots the data figure.


### Plots:

* [plot_rtp_map.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/dipping-regional/plot_rtp_map.ipynb):
  Plots the RTP anomaly and $\Gamma$ map.
* [plot_solutions.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/dipping-regional/plot_solutions.ipynb):
  Plots the results of the inversion.
* [plot_validation.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/dipping-regional/plot_validation.ipynb):
  Plots the grid of tentative values of $m_0$ and $z_0$.
* [single_vizualiation.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/dipping-regional/single_vizualiation.ipynb):
  Plots the results of the single inversion.

