# rashdf
Read data from HEC-RAS HDF files.

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


