# blobrender

**blobrender** is a Python package for turning simulations into realistic radio images. It provides tools for manipulating FITS files, generating measurement sets, and simulating telescope observations.

---

## Features

- Convert simulation outputs to FITS format
- Generate measurement sets for existing and future radio telescopes
- Predict visibilities from FITS images
- Resize and manipulate FITS files
- Estimate optically thin radio luminosity of simulations with relativistic beaming
- Containerized workflows with Docker and Singularity

---

## Installation

Clone the repository and install with pip:

```sh
git clone https://github.com/katiesavarc/blob-render.git
cd blob-render
pip install .
```

---

## Requirements

- Python 3.8+
- See `pyproject.toml` for dependencies (numpy, pyyaml, casatools, matplotlib, astropy, scipy, etc.)

---

## Command Line Tools

After installation, the following CLI tools are available:

- `blobrender.make-ms` — Generate a measurement set from a specified telescope and observing parameters
- `blobrender.fits-conversion` — Convert simulation outputs to blobrender-ready FITS images
- `blobrender.predict` — Predict visibilities using given FITS file and measurement set
- `blobrender.resize-fits` — Resize FITS images
- `blobrender.simulation-luminosity` — Estimate optically thin radio luminosity from simulation data
- `blobrender.setup-container` — Build or pull Docker/Singularity containers for workflows

Each tool can be run with `--help` for usage information.

---

## Example Usage

```sh
blobrender.simulation-luminosity
blobrender.fits-conversion 
blobrender.make-ms 
blobrender.setup-container --filetype Docker
blobrender.predict 

```

---

## Container Support

blobrender supports running workflows in Docker or Singularity containers for reproducibility. Use `blobrender.setup-container` to build or pull the required containers.

---

## License



---

## Author

Katie Savard  
katherine.savard@physics.ox.ac.uk

---

## Homepage

[https://github.com/katiesavarc/blob-render](https://github.com/katiesavarc/blob-render)
