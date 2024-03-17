#!/bin/bash

drive () {
    for i in $(seq $nprocs); do
        julia --project --check-bounds=no -t 1 -e 'include("examples/driver-exact.jl")' -- "$@" $N $nprocs $i &
    done

    wait

    julia --project --check-bounds=no -t 1 -e 'include("examples/combine.jl")' -- "$@" $N 1 # last argument sets sort=true
}

nprocs=8
N=240

touch times.txt
echo "" >> times.txt
date >> times.txt

{ time drive 2 20 12 15; } 2>&1 | grep real >> times.txt