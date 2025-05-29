# ✨ blobrender ✨

**blobrender** is a python package for turning simulations into realistic radio images.

---

## 🦶features🍻

- 🖼️ convert simulation outputs to fits format
- 📡 generate measurement sets for existing and future radio telescopes
- 🔮 predict visibilities from fits images
- ✂️ resize and manipulate fits files
- 💡 estimate optically thin radio luminosity of simulations with relativistic beaming
- 📦 containerized workflows with docker and singularity

---

## ⚡ installation

clone the repository and install with pip:

```sh
git clone https://github.com/katiesavarc/blob-render.git
cd blob-render
pip install .
```

---

## 🛠️ requirements

- 🐍 python 3.8+
- 📦 see `pyproject.toml` for dependencies (numpy, pyyaml, casatools, matplotlib, astropy, scipy, etc.)

---

## 🖥️ command line tools

after installation, the following cli tools are available:

- `blobrender.make-ms` — generate a measurement set from a specified telescope and observing parameters
- `blobrender.fits-conversion` — convert simulation outputs to blobrender-ready fits images
- `blobrender.predict` — predict visibilities using given fits file and measurement set
- `blobrender.resize-fits` — resize fits images
- `blobrender.simulation-luminosity` — estimate optically thin radio luminosity from simulation data
- `blobrender.setup-container` — build or pull docker/singularity containers for workflows

each tool can be run with `--help` for usage information.

---

## 💡 example usage

```sh
blobrender.simulation-luminosity
blobrender.fits-conversion 
blobrender.make-ms 
blobrender.setup-container --filetype docker
blobrender.predict 
```

---

## 🐳♾️  container support

blobrender supports running workflows in **docker** 🐳 or **singularity** ♾️ containers for reproducibility.  
use `blobrender.setup-container` to build or pull the required containers.

---

## 📄 license

???? not sure yet honestly

---

## 👩‍💻 author

katie savard  
katherine.savard@physics.ox.ac.uk

with much help from many others 👨‍💻

Henry Whitehead (built the DART ray-tracing module)

Ian Heywood

James Matthews

Rob Fender

Andrew Hughes

---

## 🌐 homepage

[https://github.com/katiesavarc/blob-render](https://github.com/katiesavarc/blob-render)
