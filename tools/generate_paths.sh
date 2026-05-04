#!/bin/bash

for i in $(seq 1 250); do
  printf "/home/dayan/Documents/PhD/Objective_1/data/isles_2022/sub-strokecase%04d/sub-strokecase%04d_norm.h5\n" "$i" "$i"
done > isles-22.list
