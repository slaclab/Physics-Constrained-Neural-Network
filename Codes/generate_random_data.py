#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

def generate_data(num_i, num_j, num_k, filename):
    with open(filename, 'w') as f:
        for i in range(num_i):
            for j in range(num_j):
                for k in range(num_k):
                    x = round(random.uniform(0, 1), 5)
                    y = round(random.uniform(0, 1), 5)
                    z = round(random.uniform(0, 1), 5)
                    f.write(f"{i} {j} {k} {x} {y} {z}\n")

# Specify the range for i, j, and k
num_i = 128  # You can change this as needed
num_j = 128  # You can change this as needed
num_k = 128  # You can change this as needed

# Output filename
filename = '/global/homes/j/jcurcio/perlmutter/pcnn/Volume_Data/Test128_v2_1.txt'

generate_data(num_i, num_j, num_k, filename)

print(f"Data written to {filename}")
