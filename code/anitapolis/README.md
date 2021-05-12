# Anitapolis application

Here are the notebooks, data, and results for the field application.
The results obtained in this application is in `results`.

## Data

The magnetic data presented here are a piece of the data provided by the 
Brazilian Geoligal Service (CPRM - www.cprm.gov.br), which are available 
in http://geosgb.cprm.gov.br/.

* `anitapolis_large_decimated.txt` - decimated data over a large area to perform 
regional separation
* `anitapolis_large_mag.txt` - decimated residual data over a large area.
* `anitapolis_mag.txt` - decimated residual data over a smaller area
for running the inversion algorithm.


## Notebooks

### Calculations:

* [multiple_inversion.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/anitapolis/multiple_inversion.ipynb):
  Perform the inversion for a grid of tentative values of $m_0$ and $z_0$.
* [single_inversion.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/anitapolis/single_inversion.ipynb):
  Perform the inversion for a pair of $$m_0$$ and $z_0$.
* [RTP_anomaly.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/anitapolis/RTP_anomaly.ipynb):
  Calculates the RTP anomaly and plots the figure.
* [regional_separation.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/anitapolis/regional_separation.ipynb):
  Performs the regional separation and plots the data figure.

### Plots:

* [plot_data_elevation.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/anitapolis/plot_data_elevation.ipynb):
  Plots a figure with the observed data, geometric heights of the observations, 
and the elevation of the surface on the study area (not shown in the paper).
* [plot_solutions.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/anitapolis/plot_solutions.ipynb):
  Plots the results of the inversion.
* [plot_validation.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/anitapolis/plot_validation.ipynb):
  Plots the $\Gamma$ map.
* [single_vizualiation.ipynb](http://nbviewer.jupyter.org/github/pinga-lab/magnetic-radial-inversion/blob/master/code/anitapolis/single_vizualiation.ipynb):
  Plots the results of the single inversion.

