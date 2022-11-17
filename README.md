# psc_example

An example to show why a new definition of regularity, i.e., partially strided codelet, is needed.


## Required packages

* CMake

* C/C++ compiler

* AVX2 support in the target architecture

* Intel MKL as a BLAS implementation


## Build and Run

First you should clone the repository *recursively* :

```bash
git clone --recursive  https://github.com/cheshmi/psc_example.git 
```

Then use the bash script `run_exp.sh` to build and run the code. You need to update the path to cmake and python first aand then simply run:

```bash
bash run_exp.sh
```

Upon successful completion, `psc.csv` should stay in the main directory.
