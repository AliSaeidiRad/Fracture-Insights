#!/bin/bash

for alpha in $(seq 0.0 0.05 1.0); do
    beta=$(awk "BEGIN {printf \"%.2f\", 1.0 - $alpha}")

    echo ---------------------------------------------
    echo "Running with alpha=$alpha beta=$beta"

    ./.venv/bin/python main.py \
        --dir-curves Dataset/Curves \
        --t 4 \
        --alpha "$alpha" \
        --beta "$beta" \
        --output "output/a_${alpha}_b_${beta}" \
        --sample-stl Dataset/Sample/Sample.stl
    
    echo ---------------------------------------------
done