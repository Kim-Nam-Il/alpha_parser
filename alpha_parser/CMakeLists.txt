cmake_minimum_required(VERSION 3.16)
project(alpha_parser)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable Eigen's OpenMP support to avoid conflicts
add_definitions(-DEIGEN_DONT_PARALLELIZE)

# Enable optimizations and AVX2
if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Xclang -fopenmp -I/opt/homebrew/include -I/opt/homebrew/Cellar/libomp/20.1.6/include")
    # Remove -lomp from here since we'll link the library directly
    # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/opt/homebrew/lib -lomp")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mavx2 -mfma -fopenmp")
endif()

# Find required packages
find_package(Eigen3 REQUIRED)

# Handle OpenMP differently for macOS
if(APPLE)
    # Use libomp from Homebrew
    find_library(OMP_LIB omp HINTS /opt/homebrew/lib /opt/homebrew/Cellar/libomp/20.1.6/lib)
    if(OMP_LIB)
        set(OpenMP_CXX_FOUND TRUE)
        set(OpenMP_CXX_LIBRARIES ${OMP_LIB})
        set(OpenMP_CXX_INCLUDE_DIRS "/opt/homebrew/include;/opt/homebrew/Cellar/libomp/20.1.6/include")
    endif()
else()
    find_package(OpenMP REQUIRED)
endif()

# Include directories
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(include)
if(APPLE AND OpenMP_CXX_INCLUDE_DIRS)
    include_directories(${OpenMP_CXX_INCLUDE_DIRS})
endif()
# Additional include for macOS OpenMP
if(APPLE)
    include_directories(/opt/homebrew/include)
    include_directories(/opt/homebrew/Cellar/libomp/20.1.6/include)
endif()

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.hpp")

# Create library
add_library(alpha_parser STATIC ${SOURCES} ${HEADERS})

# Create shared library for Python interface
add_library(alpha_parser_shared SHARED ${SOURCES} ${HEADERS} src/c_interface.cpp)

# Set different output names based on platform
if(APPLE)
    set_target_properties(alpha_parser_shared PROPERTIES OUTPUT_NAME "alpha_parser")
    set_target_properties(alpha_parser_shared PROPERTIES SUFFIX ".dylib")
elseif(UNIX AND NOT APPLE)
    set_target_properties(alpha_parser_shared PROPERTIES OUTPUT_NAME "alpha_parser")
    set_target_properties(alpha_parser_shared PROPERTIES SUFFIX ".so")
    # Add Linux-specific flags
    set_target_properties(alpha_parser_shared PROPERTIES 
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endif()

# Link libraries
if(APPLE)
    if(OpenMP_CXX_LIBRARIES)
        target_link_libraries(alpha_parser ${OpenMP_CXX_LIBRARIES})
        target_link_libraries(alpha_parser_shared ${OpenMP_CXX_LIBRARIES})
    endif()
else()
    target_link_libraries(alpha_parser OpenMP::OpenMP_CXX)
    target_link_libraries(alpha_parser_shared OpenMP::OpenMP_CXX)
endif()

# Create test executable
add_executable(test_alpha_parser test/test_alpha.cpp)
target_link_libraries(test_alpha_parser alpha_parser)
if(APPLE AND OpenMP_CXX_LIBRARIES)
    target_link_libraries(test_alpha_parser ${OpenMP_CXX_LIBRARIES})
endif()

# Create benchmark executable
add_executable(benchmark_alpha_parser test/benchmark_alpha.cpp)
target_link_libraries(benchmark_alpha_parser alpha_parser)
if(APPLE AND OpenMP_CXX_LIBRARIES)
    target_link_libraries(benchmark_alpha_parser ${OpenMP_CXX_LIBRARIES})
endif()

# Create matrix test executable
add_executable(matrix_test test/matrix_test.cpp)
target_link_libraries(matrix_test alpha_parser)
if(APPLE AND OpenMP_CXX_LIBRARIES)
    target_link_libraries(matrix_test ${OpenMP_CXX_LIBRARIES})
endif()

# Main executable not needed - this is a library project 