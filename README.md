# blindat

A Python library for blind analysis of measurement data in a `pandas.DataFrame()`.

Motivations for blind analysis:

"Blind Analysis" by R. MacCoun and S. Perlmutter,
[Nature volume 526, pages 187–189 (2015)](https://doi.org/10.1038/526187a).

*Do you need to blind your data?*

Possibly not. But if you expect (or want?!) a measurement to produce a certain result, blinding can remove the temptation to tune the analysis.

*Do you need a package to blind your data?*

Nope. To randomly offset the values of a column:

```python
import numpy as np
import pandas as pd

# parameters
fil = "./my_awesome_measurement.csv"
name = "A"
offset = 0.1

# load
df = pd.read_csv(fil)

# shift data values
df[name] = df[name] + offset * np.random.rand()
```

`blindat` slightly extends this simple concept using transformation rules.

### things to consider

This library is an experiment in developing a reasonable workflow for blind analysis.  It is not intended to be a universal solution for all forms of data or blind analysis techniques. 

I assume the user *wants* to avoid bias. Trust allows for a simple and reversible approach (stored data is never affected).  However, more paranoia is probably more appropriate for critical applications.

# Install

Requires python>=3.10, numpy and pandas.

Activate your Python analysis environment.  Clone the source code and `cd` into the directory.  Install using `pip`

```bash
pip install .
```

# Example

Create a `pandas.DataFrame()` with four columns of random data:


```python
# data params
COLUMNS = ["A", "B", "C", "D"]
NUM_ROWS = int(1e7)
DATA_SEED = 19421127

# generate data
np.random.seed(DATA_SEED)
data = np.random.rand(NUM_ROWS, len(COLUMNS))
df = pd.DataFrame(data, columns=COLUMNS)

df.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.519411</td>
      <td>0.030766</td>
      <td>0.064909</td>
      <td>0.930325</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.269587</td>
      <td>0.562393</td>
      <td>0.227109</td>
      <td>0.202936</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.369254</td>
      <td>0.579577</td>
      <td>0.015450</td>
      <td>0.534170</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.671910</td>
      <td>0.868601</td>
      <td>0.142738</td>
      <td>0.573955</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.903384</td>
      <td>0.921365</td>
      <td>0.019821</td>
      <td>0.263312</td>
    </tr>
  </tbody>
</table>
</div>


### `generate_rules()`

Rules are defined by a specification that describes which columns to blind using a linear transform: `blinded_data = scale * data + offset`.  

The `offset` and `scale` parameters can be fixed or randomly sampled from a given range.  Randomness ensures the transform is not known to the user.

The simplest way to specify rules is with a column name (or a list of names) and global ranges for `offset` and/or `scale`.


```python
import blindat as bd

# list of columns with global offset and scale ranges
rules = bd.generate_rules("A", offset=(10.0, 20.0), random_seed=42)
```

In this example, we want to offset the values in column 'A' by a random value selected from the range of 10 to 20.  Rules with bespoke ranges for each column can be generated using a dictionary or a list of tuples for the specification argument.

### `inspect()`

You shouldn't be looking at the rule parameters.  But maybe you have a legit reason, in which case, use `inspect()`.


```python
bd.inspect(rules)
```

    {'A': {'offset': np.float64(13.745401188473625), 'scale': np.float64(1.0)}}


It's not necessary to save the rules because they can be created reproducibly by fixing the `random_seed`.  But if you really want to store them, consider using `dill` (regular pickling doesn't work with lambda functions).  Or you could save the output of `inspect()`.

###  `blind()`


```python
# blind data
df1 = bd.blind(df, rules)
df1.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.264812</td>
      <td>0.030766</td>
      <td>0.064909</td>
      <td>0.930325</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14.014989</td>
      <td>0.562393</td>
      <td>0.227109</td>
      <td>0.202936</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14.114655</td>
      <td>0.579577</td>
      <td>0.015450</td>
      <td>0.534170</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.417311</td>
      <td>0.868601</td>
      <td>0.142738</td>
      <td>0.573955</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14.648785</td>
      <td>0.921365</td>
      <td>0.019821</td>
      <td>0.263312</td>
    </tr>
  </tbody>
</table>
</div>



### `@obfuscate`

If an experiment generates many different data files, it might be convenient to develop a custom class with methods for accessing each component. The decorator `@obfuscate` adds blinding to functions or methods that return a pandas DataFrame as the first or only result.  The method must accept the keyword argument `transform` (or `**kwargs`).


```python
from blindat import obfuscate

class MeasurementData:
    def __init__(self, path=None):
        self.path = path  # path to data directory
        self._sim()

    def _sim(self):
        np.random.seed(DATA_SEED)
        self._columns = COLUMNS
        self._data = np.random.rand(NUM_ROWS, len(self._columns))

    @obfuscate
    def load_dataframe(self, transform=None):
        df = pd.DataFrame(self._data, columns=self._columns)
        return df


# initialize
measurement = MeasurementData()

# load dataframe
measurement.load_dataframe(transform=rules).head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.264812</td>
      <td>0.030766</td>
      <td>0.064909</td>
      <td>0.930325</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14.014989</td>
      <td>0.562393</td>
      <td>0.227109</td>
      <td>0.202936</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14.114655</td>
      <td>0.579577</td>
      <td>0.015450</td>
      <td>0.534170</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.417311</td>
      <td>0.868601</td>
      <td>0.142738</td>
      <td>0.573955</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14.648785</td>
      <td>0.921365</td>
      <td>0.019821</td>
      <td>0.263312</td>
    </tr>
  </tbody>
</table>
</div>


```python
# original data
measurement.load_dataframe().head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.519411</td>
      <td>0.030766</td>
      <td>0.064909</td>
      <td>0.930325</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.269587</td>
      <td>0.562393</td>
      <td>0.227109</td>
      <td>0.202936</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.369254</td>
      <td>0.579577</td>
      <td>0.015450</td>
      <td>0.534170</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.671910</td>
      <td>0.868601</td>
      <td>0.142738</td>
      <td>0.573955</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.903384</td>
      <td>0.921365</td>
      <td>0.019821</td>
      <td>0.263312</td>
    </tr>
  </tbody>
</table>
</div>

This example requires the user to explicitly opt-in to blinding their data (zen of python #2).  

For consistency and to save the user a little effort you could include a `default_rules()` function in your data-access module.  This might be appropriate if columns with certain names always have similar values and should always be blinded.


```python
# in your data access module
DEFAULT_SPECIFICATION = {
    "A": {"offset": (10.0, 20.0), "scale": (0.9, 1.1)},
}


def default_rules(random_seed=None):
    return bd.generate_rules(DEFAULT_SPECIFICATION, random_seed=random_seed)


# in your analysis notebook
measurement.load_dataframe(transform=default_rules(42)).head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.311634</td>
      <td>0.030766</td>
      <td>0.064909</td>
      <td>0.930325</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14.039290</td>
      <td>0.562393</td>
      <td>0.227109</td>
      <td>0.202936</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14.147941</td>
      <td>0.579577</td>
      <td>0.015450</td>
      <td>0.534170</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.477879</td>
      <td>0.868601</td>
      <td>0.142738</td>
      <td>0.573955</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14.730219</td>
      <td>0.921365</td>
      <td>0.019821</td>
      <td>0.263312</td>
    </tr>
  </tbody>
</table>
</div>

Alternatively, one could hard-code rules into a data-access class.  However, forgetting about this could be disastrous! Consider using an unambiguously named subclass and/or warnings.

```python
import warnings
from blindat import blind


class BlindData(MeasurementData):
    def __init__(self, *args, random_seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._rules = default_rules(random_seed)

    def _secret_data(self):
        return super().load_dataframe()

    def load_dataframe(self):
        warnings.warn("data may be altered to mitigate experimenter bias.")
        return blind(self._secret_data(), rules=self._rules)


blind_data = BlindData(random_seed=42)

# blind by default
blind_data.load_dataframe().head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.311634</td>
      <td>0.030766</td>
      <td>0.064909</td>
      <td>0.930325</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14.039290</td>
      <td>0.562393</td>
      <td>0.227109</td>
      <td>0.202936</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14.147941</td>
      <td>0.579577</td>
      <td>0.015450</td>
      <td>0.534170</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.477879</td>
      <td>0.868601</td>
      <td>0.142738</td>
      <td>0.573955</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14.730219</td>
      <td>0.921365</td>
      <td>0.019821</td>
      <td>0.263312</td>
    </tr>
  </tbody>
</table>
</div>
