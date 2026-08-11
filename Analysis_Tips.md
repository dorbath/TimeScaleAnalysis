Many of the outlined steps are rather inuitive, but there are several steps/tools
that need additional explanation.

## Timescale analysis

First, the multi-exponential fit has some tricky regions that need special care.

### Boundaries

At the lower and upper boundary the fit will 'suddenly' have no more data points
to adjust to, which can result in a divergence of the fit or strong oscialltions.
As these regions are usually of no interest and cut-off it is only important to
ensure that it does not affect the amplitudes within the range of interest.

For the lower boundary it is recommended to put the initial fit amplitude
close to the first available data point.
Note, in case your averaged data starts from the exact same spot, the initial frame
will have a vanishing standard deviation and therefore the analysis is forced to
match this point almost perfectly resulting in large amplitudes.

In case of the upper boundary, it is sufficient to use the `tsa.extend_timeTrace()`
function which appends an additional order of magnitude allowing for a better
convergence than an sudden end. The extended value is typically derived as average
over some fraction of the final decade, which for highly non-converged data might
induce other issues that you need to treat separately.

### Rescaling of mean and sem

Some data sets contain only small values (e.g., atomic distances in [nm],
absorption spectra [cm^-1]) which are challenging for the analysis due to the small
changes. In such a case it is recommended to rescale the mean and standard error as
```python
scaling_factor = 20
tsa.options['temp_mean'] *= scaling_factor
tsa.options['temp_sem'] *= scaling_factor
...
tsa.spectrum[:, 1] /= scaling_factor
```
This will require to adjust the regularization parameter, however, the imporved
precision allows for a much better dynamical resolution.

### Fit artifacts

Even for a very good data preparation and choice of regularization, there is still
the possibility for fit artifacts.
1) In case of rapid increase/decreases, the meaningful peak is adjointed by
    another peak of opposite sign. This allows for a more precise match of the
    data, but you must be aware that the secondary peak is not physical.
2) For constant data values, the fit might osciallate between positive and
    negative values. Most often these amplitudes remain very small and thus can be
    ignored, however, it can occur that they become significant or might add up.
3) While `timescaleanalysis.timescales.derive_optimal_regularization()` provides
    a way to derive an 'optimal' regularization parameter, you might be interested
    in a highly resolved timescale and thus reduce the derived value. This will
    induce typical over-fitting artifacts (more/small unphysical peaks, shift of
    their position, splitting of timescales). Be careful and verify your analysis.
    The easiest signs are secondary peaks of opposite sign (as in point 1) and
    timescales at positions at which the data is not matching it. 

## Compuational speed up of timescale fit

### Parallelization

For data of many observables/features ($>100$),e.g., all C$_\alpha$ distances of a
protein, it is recommended to parallelize the timescale analysis fit.
You can use for instance `joblib` and wrap the for loop into a separate function as:
```python
from joblib import Parallel, delayed

def parallel_TSA(idx, tsa):
    temp_mean = utils.gaussian_smooth(tsa.data_mean[:, idxObs], 6)
    temp_sem = utils.gaussian_smooth(tsa.data_sem[:, idxObs], 6)
    ...
    return tsa.spectrum

resulting_spectrum = Parallel(n_jobs=-1)(
    delayed(parallel_TSA)(i, tsa) for i in range(tsa.data_mean.shape[1])
)
```

### Reduction of frames

Alternatively, and especially for very large number of features ($>500$) it is
recommended to decrease the number of frames on the logarithmic scale.
In such a case it is better apply the filter prior to the log-spacing in order to
retain as much information as possible. The reduced number of frames (factor of 10)
significantly speeds up the fit. Note that you will have to adjust the
regularization parameter to smaller values, roughly by the same factor as you
reduced the number of frames.