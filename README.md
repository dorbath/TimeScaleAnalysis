# Timescale analysis of time-dependent observables
The timescale analysis is a flexible data-driven analysis to reveal the strength and position in time of dynamics of a studied system.
It performs a multi-exponential fit to any observable, with non-zero amplitudes corresponding to relevant dynamics.
While this module can be used for any observable, it was design with multiple protein distances in mind.
Therefore, a "dynamical content" is defined which combines the amplitudes of each observable into a single time-resolved observable.
For more detailed information about the method, see [1](https://doi.org/10.1366/000370206776593654), [2](https://doi.org/10.1098/rstb.2017.0187) and [3](https://doi.org/10.6094/UNIFR/266874).

## Features
- Fast and simple embedding as module into your script
- Flexible adjustments to required data set and observables
- Single final time-dependent observable for the entire system

## Installation
```
pip install git+https://github.com/dorbath/TimeScaleAnalysis.git
```

### Module - As part of your python script
```python
import timescaleanalysis

# Provide path to data file(s), all are used that fulfill path/to/data*
# Load and prepare data (execute a single time)
preP = timescaleanalysis.preprocessing(data_path)
preP.generate_input_trajectories()
preP.load_trajectories()
preP.get_time_array()
preP.save_preprocessed_data()

# Perform analysis for each observable over 'fit_n_decades' decades
tsa = timescaleanalysis.timescales(preP.data_dir, fit_n_decades)
tsa.load_data()
for i in range(tsa.data_arr.shape[1]):
  tsa.options['temp_mean'] = tsa.data_mean[:, i]
  tsa.options['temp_sem'] = tsa.data_sem[:, i]
  tsa.perform_tsa(
    regPara=100,  # controls over/under fitting
    startTime=1e-1,  # first time value of fit function
  )
  timescaleanalysis.plotting.plot_TSA(
    tsa.data_mean[:, i]
    tsa.data_sem[:, i]
    tsa.spectrum,  # provide fit amplitudes
    tsa.times
  )
```

