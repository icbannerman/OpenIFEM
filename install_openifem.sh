#!/bin/bash
set -e

# Install dependencies from apt
sudo apt-get update
sudo apt-get install -y build-essential cmake git \
    mpi-default-bin mpi-default-dev \
    libdeal.ii-dev libpetsc-real-dev libslepc-real-dev \
    libp4est-dev libmetis-dev libhypre-dev

# Clone OpenIFEM source
if [ ! -d OpenIFEM-source ]; then
    git clone https://github.com/OpenIFEM/OpenIFEM.git OpenIFEM-source
fi

# Create build directory
mkdir -p OpenIFEM-build
cd OpenIFEM-build

# Configure with CMake
cmake ../OpenIFEM-source -DDEAL_II_DIR=/usr/share/cmake/deal.II

# Build
make -j$(nproc)

