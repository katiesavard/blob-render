[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)
![License](https://img.shields.io/github/license/katiesavard/blob-render)

#  blobrender 

**blobrender** is a python package for turning simulations into realistic radio images.

<p align="center">
  <img src="./cosmic_quack_of_protest.png" alt="A cosmic duck in protest, representing the spirit of blobrender" width="400"/>
  <br>
  <em>A cosmic duck in protest, representing the spirit of blobrender</em>
</p>


---

### ❗️ caution y'all❗️

This project is still **currently under development** and does contain bugs. Use at your own risk and please report unexpected behavior to [the author](mailto:katherine.savard@physics.ox.ac.uk). **Fully tested pipeline predicted to be ready for use early September 2025 -- watch this code (select 'Releases only') to be notified when the first stable version is available, as well as subsequent versions with wider capabilities**.

If you encounter problems with `pip install`-ing the project due to `casatools`, this may be due to OS incompatibility. See [this table in the casa docs](https://casadocs.readthedocs.io/en/stable/notebooks/introduction.html#Compatibility) for a list of compatible OS/python version combinations. *In the future we plan on containerizing this part of the workflow to mitigate fatal sensitivity to OS/python combinations.*

---

## Table of Contents

- [Features](#features)
- [Requirements](#%EF%B8%8F-requirements)
- [Installation](#-installation)
- [Command Line Tools](#%EF%B8%8F-command-line-tools)
- [Quickstart Example](#-quickstart-example)
- [Containers](#%EF%B8%8F--containers)
- [License](#-license)
- [Author](#-author)
- [Software roll-call](#-software-package-roll-call)
- [Homepage](#-homepage)

---

## features

- 🖼️ convert simulation outputs to fits format
- 📡 generate measurement sets for existing and future radio telescopes
- 🔮 predict visibilities from fits images
- ✂️ resize and manipulate fits files
- 💡 estimate optically thin radio luminosity of simulations with relativistic beaming
- 📦 containerized workflows with docker and singularity

---


## 🛠️ requirements

- 🐍 python 3.8+
- 📦 see `pyproject.toml` for dependencies (numpy, pyyaml, casatools, matplotlib, astropy, scipy, etc.)

---

## ⚡ installation

clone the repository and install with pip:

```sh
git clone https://github.com/katiesavarc/blob-render.git
cd blob-render
pip install .
```

You may be required to install this in a virtual environment depending on your local setup. If the above procedure causes issues, try, after cloning the repo:

```sh
virtualenv myvenv
source myvenv/bin/activate
(myvenv)$ pip install <path to checked out blob-render>
```

If you wish to install in development mode, modify the pip command to:

```sh
pip install -e <path to checked out blob-render>
```


This code is not yet available on PyPi due to it's development status but will be in the near future. 


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


For details and tutorials on how to use each of these tools, visit the [wiki page](https://github.com/katiesavarc/blob-render/wiki).


---

## 💡 quickstart example 

This software can take raw simulated data all the way to a realistic radio image with observing parameters of your choice. The first step is not necessary, and you may begin with your own image, as long as the units of the image are in emissivity (Jy) per pixel.

Default parameters for each step are stored in `configs/` as `default_simulation.yaml`, `default_MSbuilder.yaml`, and `default_prediction.yaml`. Command-line arguments overwrite these defaults. 

See [wiki page](https://github.com/katiesavard/blob-render/wiki) for more details on these steps, or follow the [tutorial section of the wiki](https://github.com/katiesavard/blob-render/wiki/Tutorial) with provided test data.  

Most of these steps can be feasibly performed on the average desktop computer, but the prediction step can be particularly intesive if measurement sets are large and we therefore recommend that this be installed and run on a computing cluster / HPC environment.


1) **Convert raw simulation data to an estimate of optically thin synchrotron emissivity:**

    ‼️ *this step is still highly specific to PLUTO .flt and .dbl files in cylindrical coordinates.* ‼️


    ‼️ *Not fit for general use (yet), use at your own caution* ‼️

    Store simulation files in `sim_data` in a folder with your simulation name. Update `configs/default_simulation.yaml` with appropriate parameters or specify with command-line flags.

    ```sh
    blobrender.simulation-luminosity
    ```

2) **Format an image or numpy file of simulation emissivity into a FITS file (necessary format for prediction step):**

    Image should be saved in `sim_data`. Update `configs/default_simulation.yaml` with appropriate parameters or specify with command-line flags.

    ```sh
    blobrender.fits-conversion
    ```

3) **Generate a measurement set from a specified telescope and observing parameters:**

    This step in independent from the previous two. Here is where you will choose the observing parameters of your simulated observation, as well as which telescope you wish to observe with. 
    Currently only the following telescopes are supported:

    - eMERLIN
    - MeerKAT
    - SKA-Mid

    Update `configs/default_MSbuilder.yaml` with appropriate parameters or specify with command-line flags.

    ```sh
    blobrender.make-ms
    ```

4) **Set up your container environment (optional, but recommended):**

    The next prediction step we recommend to do in a container. This is because the prediction requires the use of the [wsclean](https://gitlab.com/aroffringa/wsclean) software, which has specific dependencies and will only run on certain operating systems. 

    This workflow supports both Docker and Singularity containers, with a slight preference for Singularity due to it's adaptation to HPC environments.

    ```sh
    blobrender.setup-container --filetype Singularity --localfile 
    ```
    If you have wsclean installed locally and wish to run prediction without a container (at your own risk, dependencies are listed in the `containers/blobrender-dockerfile`), simply run the setup as: 

    ```sh
    blobrender.setup-container --nocontainer
    ```

    See [container support](#containers) for how to set up Docker or Singularity for the first time, and generally the [wiki page](https://github.com/katiesavard/blob-render/wiki) for more in-depth information. 


5) **Predict visibilities using the FITS file and measurement set:**

    This is the *beefiest* step. For large measurement sets we recommend performing this step on a computing cluster with more RAM than a regular desktop environment.

    This step will predict what your provided FITS file (located in `sim_data/`) image looks like with the observing parameters specified in the measurement set (located in the working directory) created from the `blobrender.make-ms` step.
    
    Update `configs/default_prediciton.yaml` with appropriate parameters or specify with command-line flags.

    ```sh
    blobrender.predict
    ```



---

## 🐳♾️  Containers

**blobrender** runs the prediction workflows in **Docker** 🐳 or **Singularity** ♾️ containers for reproducibility and ease of installation / managing dependencies.

**See [Docker Docs](https://docs.docker.com/engine/install/) for installing Docker (note that on Mac OS the desktop app must be running to perform builds) and [Singularity Docs](https://docs.sylabs.io/guides/3.0/user-guide/installation.html) for installing Singularity (not available on Mac OS, and also may be known as [Apptainer](https://apptainer.org/docs/user/latest/singularity_compatibility.html) on newer systems but retains the same syntax).**


Use `blobrender.setup-container` to build or pull the required containers.

Docker and Singularity containers can be built from the files provided in `containers/` by using the `--local-file` flag. This is recommended, as **blobrender** is tested in these containers. 

If you encounter problems building locally (conflicts with user privileges), you can choose to pull a pre-built image from DockerHub where we recommend you use the [wsclean container provided by stimela](https://hub.docker.com/layers/stimela/wsclean/1.6.3/images/sha256-15718fd7303ba215829f3eca0bc04e6e28af23170a5acae6b7d6a316b2eb4bce) (Makhathini et al. 2018). This is done automatically if the `--local-file` flag is omitted. 

As mentioned in the [quickstart example](#quickstart-example), we recommend using Singularity as it plays better with HPC environments. However, Singularity is not supported on Mac OS, hence why we also provide the option to use Docker (although, we risk sounding like a broken record, but for most use cases this software is best used on HPC due to RAM requirements).

Lastly, if using Singularity, it is often the case on clusters that one does not have root privileges (no `sudo` rights) and thus can not build containers from scratch. For this reason we provided the functionality to pull a pre-built container from DockerHub, but we also provide the option to build the containers provided with blobrender (recommended option) on a remote server know as [Sylabs](https://cloud.sylabs.io/builder). 

> "Using the Remote Builder, you can easily and securely create containers for your applications without special privileges or set up in your local environment. The Remote Builder can securely build a container for you from a definition file entered here or via the Singularity CLI."
> – [Sylabs](https://cloud.sylabs.io/builder)

To enable this functionality, you must first create an account with [Sylabs](https://cloud.sylabs.io/builder). Once you have an account, ensure you are logged in from your command-line environment by running

```sh
singularity remote login
```
This may prompt you to create an Access Token which can be obtained via the Sylabs website and should be saved in your `$HOME/.singularity/remote.yaml`, if not automatically created. Once this is set up, you can check your login status with:
```sh
singularity remote status
```

You are now clear to run the `blobrender.setup-container` command with the `--remote-build` flag, noting that this option is only available when `--filetype Singularity`. 



---

## 📄 license

MIT

---

## 👩‍💻 author

katie savard  
katherine.savard@physics.ox.ac.uk

with much help from many others 👨‍💻

Henry Whitehead (built the DART ray-tracing module)

Ian Heywood (see [oxkat](https://github.com/IanHeywood/oxkat) software, a heavy inspiration)

James Matthews

Rob Fender

Andrew Hughes (invaluable help with containerizing the wsclean installation)

---

## 📦 Software Package Roll-Call

Nothing would work without these fantastic packages. This is basically a duct-taped package of packages. 

| Package/Tool      | Purpose/Role                                      | Link/Source                                      | Author               |
|-------------------|---------------------------------------------------|--------------------------------------------------|---------------------------------|
| **numpy**         | Numerical operations and array handling           | [numpy.org](https://numpy.org/)                  | Travis Oliphant et al.          |
| **pyyaml**        | YAML config parsing                               | [pyyaml.org](https://pyyaml.org/)                | Kirill Simonov                  |
| **casatools**     | Radio astronomy data handling                     | [CASA](https://casa.nrao.edu/)                   | NRAO/CASA Team                  |
| **matplotlib**    | Plotting and visualization                        | [matplotlib.org](https://matplotlib.org/)        | John D. Hunter et al.           |
| **astropy**       | Astronomy utilities and FITS I/O                  | [astropy.org](https://www.astropy.org/)          | Astropy Collaboration           |
| **scipy**         | Scientific computing                              | [scipy.org](https://scipy.org/)                  | SciPy Developers                 |
| **pyPLUTO**       | PLUTO simulation data reading                     | [pyPLUTO GitLab](https://gitlab.mpcdf.mpg.de/sdoetsch/pypluto) | S. Doetsch, PLUTO Team          |
| **pyrap**         | Measurement set handling                          | [pyrap GitHub](https://github.com/casacore/python-casacore)    | CASAcore Developers             |
| **pillow**        | Image file reading/writing                        | [python-pillow.org](https://python-pillow.org/)  | Alex Clark, Pillow Contributors |
| **numba**         | Fast numerical code via JIT compilation           | [numba.pydata.org](https://numba.pydata.org/)    | Anaconda, Inc.                  |
| **docker**        | Containerization  | [docker.com](https://www.docker.com/)            | Docker, Inc.                    |
| **singularity**   | Containerization   | [apptainer.org](https://apptainer.org/)          | Apptainer/Singularity Team      |
| **wsclean**       | Radio interferometric image synthesis (cleaning)   | [wsclean GitLab](https://gitlab.com/aroffringa/wsclean) | André Offringa et al.           |

---

*For more details on dependencies, see `pyproject.toml` or the [wiki page](https://github.com/katiesavard/blob-render/wiki).*

---


## Testing

There is currently no testing happening. Beware. 
Coming soon. Report any bugs to [the author](mailto:katherine.savard@physics.ox.ac.uk) or create a GitHub issue. 


## 🌐 homepage

[https://github.com/katiesavarc/blob-render](https://github.com/katiesavard/blob-render)
