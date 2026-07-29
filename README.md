# Scatter Analysis

A Python package providing tools for the analysis of X-ray and neutron scattering data (SAXS, SANS, and GID).

## Installation

```bash
pip install scatter-analysis
```

## Features

- **GID**: Analysis of Grazing Incidence Diffraction data. Calculates crystallographic parameters based on scattering vectors.
- **SAS**: Analysis of Small Angle Scattering data, including Small Angle X-ray Scattering (SAXS) and Small Angle Neutron Scattering (SANS) with support for plotting and data loading.

## Documentation

Documentation is available at [scatter-analysis docs](https://github.com/user/scatter-analysis)

## Usage Example

```python
from gid import calculate_cell_parameters

# Calculate parameters based on scattering vectors
params = calculate_cell_parameters(qxy1=1.0, qxy2=1.0, qxy3=1.0, qz1=0.1, qz2=0.1, qz3=0.1)
print(params)
```
