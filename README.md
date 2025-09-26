# EFFICIENT BATCHED CPU/GPU IMPLEMENTATION OF ORTHOGONAL MATCHINGPURSUIT FOR PYTHON
Orthogonal Matching Pursuit, implemented using BLAS (CPU) and PyTorch (GPU).

Our implementations vastly outperform those in Scikit-Learn, with the PyTorch version on GPU being over 100 times faster.

A demo along with the implementation can be found in [this Colab](https://colab.research.google.com/drive/1BwqjGQC5XfaRiTUxit-afW0vg6ezjsh5?usp=sharing), also available in this repo as `Quickstart.ipynb`.

Associated Paper feat. Sebastian Praesius!: [Efficient Batched CPU/GPU OMP](https://github.com/Ariel5/omp-parallel-gpu-python/blob/main/results/Compressed_Sensing_Report.pdf).


# Currently only works with py39!


## Installation

0. Ensure you have Python Development tools installed

```
sudo apt-get update
sudo apt-get install build-essential python3.9-dev   # Replace with specific python version you are using in your venv
```

1. You must compile the Cython functions:

```
python3 ./src/cython/setup.py build_ext --inplace
```




### Notes

We do not yet provide a MATLAB version of our code, but a good, efficient version can be found [here](https://github.com/zhuhufei/OMP/blob/master/codeAug2020.m) - though it is lacking many of our optimizations
