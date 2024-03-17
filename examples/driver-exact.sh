#!/bin/bash

nprocs=8
N=240

for i in $(seq $nprocs); do
    julia --project --check-bounds=no -t 1 -e 'include("examples/driver-exact.jl")' -- 2 20 12 15 $N $nprocs $i &
done

wait

julia --project --check-bounds=no -t 1 -e 'include("examples/combine.jl")' -- 2 20 12 15 $N 1