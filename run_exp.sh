#!/bin/bash


BINDIR=./build/
DATADIR=./data/
PYTHON=/Users/kazem/anaconda3/envs/nano/bin/python3
CMAKE=/Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake
CMAKE=cmake
PYTHON=/home/kazem/anaconda3/envs/codegen/bin/python

mkdir ${BINDIR}
cd ${BINDIR}
cmake -DCMAKE_BUILD_TYPE=Release ..
make psc_example

cd ..
mkdir ${DATADIR}

echo "Matrix Name,Rows,Columns,NNZ,Scalarized,BLAS,PSC G,PSC 2D,PSC 2D-4,BLAS Check,PSC G Check,PSC 2D Check,PSC 2D-4 Check," > ./psc.csv
for i0 in 4 8 16 32 64; do
  for blk_no in 64 128 256 1024 2048 4096 8192; do
    for max_bs in 8 10 12 14 16 18 20 30 40 50 60 70; do
      ${PYTHON}  pattern_gen.py ${blk_no} ${i0} ${max_bs} ${DATADIR}
      ${BINDIR}/psc_example ${DATADIR}/mat_${blk_no}_${i0}_${max_bs}.mtx >> ./psc.csv
    done
  done
done


#plotting
${PYTHON}  plot.py ./psc.csv