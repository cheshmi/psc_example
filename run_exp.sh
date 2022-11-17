#!/bin/bash


BINDIR=./build/
DATADIR=./data/

mkdir ${BINDIR}
cd ${BINDIR}
/Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -DCMAKE_BUILD_TYPE=Release ..
make psc_example

cd ..
mkdir ${DATADIR}

for i0 in 4 8 16 32 64; do
  for blk_no in 4 8 16 32 64 128 256 1024; do
    for max_bs in 2 4 6 8; do
      /Users/kazem/anaconda3/envs/nano/bin/python3 pattern_gen.py ${blk_no} ${i0} ${max_bs} ${DATADIR}
      ${BINDIR}/psc_example ${DATADIR}/mat_${blk_no}_${i0}_${max_bs}.mtx
    done
  done
done


