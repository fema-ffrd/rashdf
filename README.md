# rashdf
[![CI](https://github.com/fema-ffrd/rashdf/actions/workflows/continuous-integration.yml/badge.svg?branch=main)](https://github.com/fema-ffrd/rashdf/actions/workflows/continuous-integration.yml)
[![Release](https://github.com/fema-ffrd/rashdf/actions/workflows/release.yml/badge.svg)](https://github.com/fema-ffrd/rashdf/actions/workflows/release.yml)
[![PyPI version](https://badge.fury.io/py/rashdf.svg)](https://badge.fury.io/py/rashdf)

Read data from [HEC-RAS](https://www.hec.usace.army.mil/software/hec-ras/) [HDF](https://github.com/HDFGroup/hdf5) files.

*Pronunciation: `raz·aitch·dee·eff`*

## Install
A prerelease version of `rashdf` is available from PyPI:
```bash
$ pip install rashdf=0.1.0b1
```

## Developer Setup
Create a virtual environment in the project directory:
```
$ python -m venv venv-rashdf
```

Activate the virtual environment:
```
$ source ./venv/bin/activate
(venv-rashdf) $
```

Install dev dependencies:
```
(venv-rashdf) $ pip install ".[dev]"
```

Install git hook scripts (used for automatic liniting/formatting)
```
(venv-rashdf) $ pre-commit install
```

With the virtual environment activated, run the tests:
```
(venv-rashdf) $ pytest
```


