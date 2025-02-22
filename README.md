# blindat

A Python library for blind analysis of measurement data in a `pandas.DataFrame()`.

*Do you need to blind your data?*

Maybe not. But if you expect (or want?!) a measurement to produce a certain result, blinding can remove the temptation to *tune* your analysis.

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

I assume the user *wants* to avoid bias. Trust allows for a simple and reversible approach (stored data is never altered).  More paranoia is more appropriate for critical applications.

# Install

Requires python>=3.10, numpy and pandas.

Activate your Python analysis environment.  Clone the source code and `cd` into the directory.  Install using `pip`

```bash
pip install .
```

# Usage

Blind analysis can be as simple as applying an unknown transform to an appropriate column of data (e.g., microwave frequency).  

In this example, a random offset is selected from the range of 10 to 20 and added to column `A` of the pandas DataFrame, `df`.

```python
import blindat as bd

rules = bd.generate_rules("A", offset=(10.0, 20.0), random_seed=42)

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

The `@obfuscate` decorator can be used to wrap methods that load a `pandas.Dataframe` with the `blind()` function. See docs for examples.
