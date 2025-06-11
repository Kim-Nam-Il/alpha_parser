#!/bin/bash

# Build script for Linux systems
echo "Building Alpha Parser for Linux..."

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Warning: This script is designed for Linux systems"
    echo "Current OS: $OSTYPE"
fi

# Install dependencies (Ubuntu/Debian)
if command -v apt-get &> /dev/null; then
    echo "Installing dependencies with apt-get..."
    sudo apt-get update
    sudo apt-get install -y build-essential cmake libeigen3-dev libomp-dev
fi

# Install dependencies (CentOS/RHEL/Fedora)
if command -v yum &> /dev/null; then
    echo "Installing dependencies with yum..."
    sudo yum install -y gcc-c++ cmake eigen3-devel libomp-devel
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_FLAGS="-O3 -march=native -fopenmp" \
         -DCMAKE_POSITION_INDEPENDENT_CODE=ON

# Build
echo "Building..."
make -j$(nproc)

# Check if .so file was created
if [ -f "libalpha_parser.so" ]; then
    echo "✅ Successfully built libalpha_parser.so"
    ls -la libalpha_parser.so
else
    echo "❌ Failed to build libalpha_parser.so"
    exit 1
fi

echo "Build completed!"
echo "The .so file is located at: $(pwd)/libalpha_parser.so" 