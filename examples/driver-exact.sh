#!/bin/zsh

nprocs=8

for i in {1..$nprocs}
do
    julia --project --check-bounds=no -t 1 -e 'include("examples/driver-exact.jl")' -- 2 20 12 15 320 $nprocs $i &
done

wait

julia --project --check-bounds=no -t 1 -e 'include("examples/combine.jl")' -- 2 20 12 15 320 1