# ============================================================
# PH Power System Framework — System Dependencies
# ============================================================
#
# This project generates and compiles C++ code at runtime.
# The following system-level packages are required in addition
# to the Python packages listed in requirements.txt.
#
# ----------------------------------------------------------
# 1. C++ Compiler (required)
# ----------------------------------------------------------
#   The framework generates C++ simulation kernels and compiles
#   them with g++.  Any C++17-capable compiler works.
#
#   Ubuntu / Debian:
#       sudo apt-get install g++
#
#   Fedora / RHEL:
#       sudo dnf install gcc-c++
#
#   macOS (Xcode command-line tools):
#       xcode-select --install
#
# ----------------------------------------------------------
# 2. SUNDIALS (optional — for IDA solver)
# ----------------------------------------------------------
#   SUNDIALS is a C/C++ library from Lawrence Livermore
#   National Laboratory providing production-grade DAE / ODE
#   solvers.  The framework uses the IDA solver (variable-order
#   BDF with adaptive time stepping) as an alternative to the
#   built-in BDF-1 Newton solver.
#
#   SUNDIALS is optional: the built-in BDF-1 solver works
#   without it.  IDA is recommended for larger systems
#   (IEEE 14-bus, 39-bus, etc.) where adaptive stepping and
#   higher-order methods improve accuracy and performance.
#
#   Ubuntu / Debian:
#       sudo apt-get install libsundials-dev
#
#   Fedora / RHEL:
#       sudo dnf install sundials-devel
#
#   macOS (Homebrew):
#       brew install sundials
#
#   From source (any platform):
#       https://computing.llnl.gov/projects/sundials
#
#   The IDA solver links against these libraries:
#       -lsundials_ida
#       -lsundials_nvecserial
#       -lsundials_sunlinsoldense
#       -lsundials_sunmatrixdense
#
# ----------------------------------------------------------
# Quick-start (Ubuntu / Debian)
# ----------------------------------------------------------
#   sudo apt-get install g++ libsundials-dev
#   pip install -r requirements.txt
#
# ----------------------------------------------------------
# Tested Versions
# ----------------------------------------------------------
#   Python     : 3.12
#   g++        : 13.3 (Ubuntu 24.04)
#   SUNDIALS   : 6.4.1
#   numpy      : 1.26
#   pandas     : 2.1
#   matplotlib : 3.6
#   scipy      : 1.11
#   networkx   : 3.3
# ============================================================
