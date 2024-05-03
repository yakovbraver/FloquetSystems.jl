#!/bin/bash
# A script for calculating quasienergy spectra in batches.

printf "\n########\nScript launched on $(date)\n" >> times.txt

### Exact

PARAMS=(
"2 20 0    30   294"
"2 20 5    8    294"
"2 20 10   15   588" 
"2 20 25   27.5 294"

"2 30 0    45   294"
"2 30 7.5  12   294"
"2 30 18   22   588"
"2 30 38.5 41   294"
)

for SET in "${PARAMS[@]}"; do
    printf "${SET}: " >> times.txt
	/usr/bin/time -f "%E" -a -o times.txt julia --project --check-bounds=no -p 21 -e 'include("examples/driver-exact.jl")' -- ${SET}
done

### (E)DPT

PARAMS=(
"2 20 0    30   320 3"
"2 20 5    8    320 3"
"2 20 10   15   640 3" 
"2 20 25   27.5 320 3"

"2 30 0    45   320 3"
"2 30 7.5  12   320 3"
"2 30 18   22   640 3"
"2 30 38.5 41   320 3"

"2 20 5    8    320 3 1 3"
"2 20 10   15   640 3 2 3" 
"2 20 25   27.5 320 3 4 3"

"2 30 7.5  12   320 3 1 3"
"2 30 18   22   640 3 2 3"
"2 30 38.5 41   320 3 4 3"
)

for SET in "${PARAMS[@]}"; do
    printf "${SET}: " >> times.txt
	/usr/bin/time -f "%E" -a -o times.txt julia --project --check-bounds=no -t 64 -e 'include("examples/driver-dpt.jl")' -- ${SET}
done
