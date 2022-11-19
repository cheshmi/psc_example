# psc_example

An example to show why a new definition of regularity, i.e., partially strided codelet, is needed. For details, please visit [this blog post](https://blog.cheshmi.cc/redefining-regularity-with-PSC.html).

## Required packages

* CMake
* C/C++ compiler
* AVX2 support in the target architecture
* Intel MKL as a BLAS implementation
* Python for data generation and plotting (Numpy, ScciPy, Matplotlib, Pandas)

## Build and Run

First you should clone the repository *recursively* :

```bash
git clone --recursive  https://github.com/cheshmi/psc_example.git 
```

Then use the bash script `run_exp.sh` to build and run the code. You need to update the path to cmake and python first aand then simply run:

```bash
bash run_exp.sh
```

Upon successful completion, `psc.csv` should be generated in the main directory. It also generates a plot as a png file 
(`psc_plot.png`).
