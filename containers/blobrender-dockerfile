FROM ubuntu:22.04

# Set timezone and noninteractive mode
ENV TZ=Etc/UTC
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y software-properties-common && \
    apt-get update -y && \
    apt-get upgrade -y && \
    ln -fs /usr/share/zoneinfo/Canada/Mountain /etc/localtime && \
    apt-get install -y tzdata && \
    apt-get install -y casacore-dev cmake g++ \
        git pkg-config libblas-dev \
        libboost-date-time-dev libboost-filesystem-dev \
        libboost-program-options-dev libboost-system-dev \
        libcfitsio-dev libfftw3-dev libgsl-dev \
        libhdf5-dev liblapack-dev libopenmpi-dev \
        libpython3-dev pkg-config && \
    git clone -b v3.5 https://gitlab.com/aroffringa/wsclean.git && \
    cd wsclean && \
    mkdir -p build && \
    cd build && \
    cmake ../ -DCMAKE_PREFIX_PATH="../../EveryBeam/" && \
    make -j 4 && \
    make install && \
    cd ../../ && \
    rm -rf wsclean

# Set working directory (optional)
WORKDIR /workspace

# Set entrypoint (optional)
# ENTRYPOINT ["/bin/bash"]