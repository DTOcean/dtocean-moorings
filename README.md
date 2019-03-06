[![appveyor](https://ci.appveyor.com/api/projects/status/github/DTOcean/dtocean-moorings?branch=master&svg=true)](https://ci.appveyor.com/project/DTOcean/dtocean-moorings)
[![codecov](https://codecov.io/gh/DTOcean/dtocean-moorings/branch/master/graph/badge.svg)](https://codecov.io/gh/DTOcean/dtocean-moorings)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/bb34506cc82f4df883178a6e64619eaf)](https://www.codacy.com/project/H0R5E/dtocean-moorings/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=DTOcean/dtocean-moorings&amp;utm_campaign=Badge_Grade_Dashboard&amp;branchId=8410911)
[![release](https://img.shields.io/github/release/DTOcean/dtocean-moorings.svg)](https://github.com/DTOcean/dtocean-moorings/releases/latest)

# DTOcean Mooring and Foundations Module

This package provides the Mooring and Foundations design module for the DTOcean 
tools. It can design the required station keeping for array of fixed or floating 
wave or tidal ocean energy converters (constrained by the environment and 
device design), and calculate the cost. It optimises the design for minimum 
cost given a specified safety factor. 

See [dtocean-app](https://github.com/DTOcean/dtocean-app) or [dtocean-core](
https://github.com/DTOcean/dtocean-app) to use this package within the DTOcean
ecosystem.

* For python 2.7 only.

## Installation

Installation and development of dtocean-moorings uses the [Anaconda 
Distribution](https://www.anaconda.com/distribution/) (Python 2.7)

### Conda Package

To install:

```
$ conda install -c dataonlygreater dtocean-moorings
```

### Source Code

Conda can be used to install dependencies into a dedicated environment from
the source code root directory:

```
conda create -n _dtocean-moor python=2.7 pip
```

Activate the environment, then copy the `.condrc` file to store installation  
channels:

```
$ conda activate _dtocean-moor
$ copy .condarc %CONDA_PREFIX%
```

Install [polite](https://github.com/DTOcean/polite) into the environment. For 
example, if installing it from source:

```
$ cd \\path\\to\\polite
$ conda install --file requirements-conda-dev.txt
$ pip install -e .
```

Finally, install dtocean-moorings and its dependencies using conda and pip:

```
$ cd \\path\\to\\dtocean-moorings
$ conda install --file requirements-conda-dev.txt
$ pip install -e .
```

To deactivate the conda environment:

```
$ conda deactivate
```

### Tests

A test suite is provided with the source code that uses [pytest](
https://docs.pytest.org).

If not already active, activate the conda environment set up in the [Source 
Code](#source-code) section:

```
$ conda activate _dtocean-moor
```

Install packages required for testing to the environment (one time only):

```
$ conda install -y pytest pytest-mock
```

Run the tests:

``` 
$ py.test tests
```

### Uninstall

To uninstall the conda package:

```
$ conda remove dtocean-moorings
```

To uninstall the source code and its conda environment:

```
$ conda remove --name _dtocean-moor --all
```

## Usage

Example scripts are available in the "examples" folder of the source code.

```
cd examples
python fairhead.py
```

## Component database keys and values

The keys of the `compdict` argument to the Variables class have the following
meanings:

* item1: Mooring or foundation system
* item2: Name
* item3: Subname
* item4: [Material, Grade, Colour]
* item5: Strength and mechanical properties. For piles [yield stress, Young's 
         modulus]. For ropes [minimum break load, [load, axial stiffness]]. For 
         cables [minimum break load, minimum bend radius]. All other 
         components: [minimum break load, axial stiffness]
* item6: Size. For piles/suction caissons [diameter, thickness]. For anchors 
         [width, depth, height, connecting size]. For chains, forerunner  
         assemblies, shackles and swivels [diameter, connecting length]. For 
         ropes and cables [outer diameter]
* item7: Mass. For piles/suction caissons, chains, ropes, cables and forerunner 
         assemblies [dry mass per unit length, wet mass per unit length]. For 
         anchors, shackles and swivels [dry unit mass, wet unit mass]
* item8: Environmental impact
* item9: Anchor coefficients
* item10: Failure rates
* item11: Cost

## Contributing

Pull requests are welcome. For major changes, please open an issue first to
discuss what you would like to change.

See [this blog post](
https://www.dataonlygreater.com/latest/professional/2017/03/09/dtocean-development-change-management/)
for information regarding development of the DTOcean ecosystem.

Please make sure to update tests as appropriate.

## Credits

This package was initially created as part of the [EU DTOcean project](
https://www.dtoceanplus.eu/About-DTOceanPlus/History) by:

 * Sam Weller at [the University of Exeter](https://www.exeter.ac.uk/)
 * Jon Hardwick at [the University of Exeter](https://www.exeter.ac.uk/)
 * Mathew Topper at [TECNALIA](https://www.tecnalia.com)

It is now maintained by Mathew Topper at [Data Only Greater](
https://www.dataonlygreater.com/).

## License

[GPL-3.0](https://choosealicense.com/licenses/gpl-3.0/)
