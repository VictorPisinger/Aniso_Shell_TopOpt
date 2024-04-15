#!/bin/bash

for file in ./*.e; do pvbatch ./exo2ply.py -f $file -o "${file%.e}.ply"; done



