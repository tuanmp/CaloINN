#!/bin/bash
# gpu_requirement="cpu"
gpu_requirement="gpu"
salloc -A m2616 -C $gpu_requirement -q interactive --nodes 1 --ntasks-per-node 1 --cpus-per-task 32 --time 04:00:00 --signal=SIGUSR1@180 #--image=tuanpham1503/torch_conda:0.4
