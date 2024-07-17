#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# -----------------------------------------------

PATH_TO_VOLUME_DATA = '/global/homes/j/jcurcio/perlmutter/pcnn/Volume_Data/'

# Give full name of filename. Any timestep is fine since the code removes the last 4 chars anyways
B_VOLUME_DATA = 'Test128_v2_0.txt'
E_VOLUME_DATA = 'Test128_v2_0.txt'
J_VOLUME_DATA = 'Test128_v2_0.txt'
Q_VOLUME_DATA = 'Test128_v2_0.txt'
Qnz_VOLUME_DATA = 'Test128_v2_0.txt'

# 1 timestep would be the 0th step. 2 would be the 0th and 1st
n_timesteps = 2
n_pixels = 128

# -----------------------------------------------

def parse_data(filename, variable_name):
    # Load data from file
    data = np.loadtxt(filename)

    if variable_name in ['B', 'E', 'J']:
        # Extract Bx, By, Bz directly from columns for B, E, and J variables
        x = data[:, 3]
        y = data[:, 4]
        z = data[:, 5]

        # Reshape Bx, By, Bz into n_pixels x n_pixels x n_pixels arrays
        x_3d = x.reshape((n_pixels, n_pixels, n_pixels))
        y_3d = y.reshape((n_pixels, n_pixels, n_pixels))
        z_3d = z.reshape((n_pixels, n_pixels, n_pixels))

        return x_3d, y_3d, z_3d

    else:
        # For Q and Qnz, only consider the 4th column
        x = data[:, 3]
        x_3d = x.reshape((n_pixels, n_pixels, n_pixels))
        return x_3d, None, None


# Function to process data and save
def process_data(variable_data, variable_name, components_names, step, save_max=False):
    x, y, z = parse_data(PATH_TO_VOLUME_DATA + variable_data[:-5] + str(step) + '.txt', variable_name)

    print(f'Shape of x for {variable_name}:', x.shape)
    if y is not None:
        print(f'Shape of y for {variable_name}:', y.shape)
    if z is not None:
        print(f'Shape of z for {variable_name}:', z.shape)

    if components_names[0] == '':
        np.save(PATH_TO_VOLUME_DATA + f'{variable_name}_3D_vol_{n_pixels}_{step}.npy', x)
    else:
        np.save(PATH_TO_VOLUME_DATA + f'{variable_name}{components_names[0]}_3D_vol_{n_pixels}_{step}.npy', x)
        np.save(PATH_TO_VOLUME_DATA + f'{variable_name}{components_names[1]}_3D_vol_{n_pixels}_{step}.npy', y)
        np.save(PATH_TO_VOLUME_DATA + f'{variable_name}{components_names[2]}_3D_vol_{n_pixels}_{step}.npy', z)

    if save_max:
        if y is None and z is None:
            max_value = np.max(np.abs(x))
        else:
            max_value = max(np.max(np.abs(x)), np.max(np.abs(y)), np.max(np.abs(z)))
        return max_value
    else:
        return None


# Variables to store overall max values
B_max_overall = float('-inf')
J_max_overall = float('-inf')

# Process each timestep and compute max values
for step in range(n_timesteps):
    B_max = process_data(B_VOLUME_DATA, 'B', ['x', 'y', 'z'], step, save_max=True)
    E_max = process_data(E_VOLUME_DATA, 'E', ['x', 'y', 'z'], step)
    J_max = process_data(J_VOLUME_DATA, 'J', ['x', 'y', 'z'], step, save_max=True)
    Q_max = process_data(Q_VOLUME_DATA, 'Q', [''], step)
    Qnz_max = process_data(Qnz_VOLUME_DATA, 'Qnz', [''], step)

    # Update overall max values
    if B_max > B_max_overall:
        B_max_overall = B_max
    if J_max > J_max_overall:
        J_max_overall = J_max

# Save max values as .npy for normalization
np.save(PATH_TO_VOLUME_DATA + 'Bxyz_max.npy', np.array(B_max_overall))
np.save(PATH_TO_VOLUME_DATA + 'J_max_max_all_128.npy', np.array(J_max_overall))
