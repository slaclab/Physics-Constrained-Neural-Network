#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# -----------------------------------------------

PATH_TO_VOLUME_DATA='/global/homes/j/jcurcio/perlmutter/pcnn/Volume_Data/'

# Give full name of filename. Any timestep is fine since the code removes the last 4 chars anyways
B_VOLUME_DATA='TestVariableA_0.txt'
E_VOLUME_DATA='TestVariableA_0.txt'
J_VOLUME_DATA='TestVariableA_0.txt'

# 1 timestep would be the 0th step. 2 would be the 0th and 1st
n_timesteps = 1
n_pixels = 10

# -----------------------------------------------
# Magnetic | B

for step in range(n_timesteps):

    data = np.loadtxt(PATH_TO_VOLUME_DATA + B_VOLUME_DATA[:-5] + str(step) + '.txt', skiprows=3)

    Bx = np.zeros([n_timesteps, n_pixels, n_pixels, n_pixels, 1])
    By = np.zeros([n_timesteps, n_pixels, n_pixels, n_pixels, 1])
    Bz = np.zeros([n_timesteps, n_pixels, n_pixels, n_pixels, 1])

    # Convert coordinates to indices and populate the Bx array
    for row in data:
        x, y, z, bx, by, bz = row
        i = int(np.round(x / n_pixels * (n_pixels - 1)))
        j = int(np.round(y / n_pixels * (n_pixels - 1)))
        k = int(np.round(z / n_pixels * (n_pixels - 1)))

        # Populate data at x,y,z
        Bx[step, i, j, k, 0] = bx
        By[step, i, j, k, 0] = by
        Bz[step, i, j, k, 0] = bz

    # Print the number of non-zero elements
    print("Number of non-zero values in Bx:", np.count_nonzero(Bx))
    print("Number of non-zero values in By:", np.count_nonzero(By))
    print("Number of non-zero values in Bz:", np.count_nonzero(Bz))

    np.save(PATH_TO_VOLUME_DATA+'Bx_3D_vol_' + str(n_pixels) + '_' + str(step) + '.npy', Bx)
    np.save(PATH_TO_VOLUME_DATA+'By_3D_vol_' + str(n_pixels) + '_' + str(step) + '.npy', By)
    np.save(PATH_TO_VOLUME_DATA+'Bz_3D_vol_' + str(n_pixels) + '_' + str(step) + '.npy', Bz)

# -----------------------------------------------
# Electric | E

for step in range(n_timesteps):

    data = np.loadtxt(PATH_TO_VOLUME_DATA + E_VOLUME_DATA[:-5] + str(step) + '.txt', skiprows=3)

    Ex = np.zeros([2, n_pixels, n_pixels, n_pixels, 1])
    Ey = np.zeros([2, n_pixels, n_pixels, n_pixels, 1])
    Ez = np.zeros([2, n_pixels, n_pixels, n_pixels, 1])

    # Convert coordinates to indices and populate the Ex array
    for row in data:
        x, y, z, ex, ey, ez = row
        i = int(np.round(x / n_pixels * (n_pixels - 1)))
        j = int(np.round(y / n_pixels * (n_pixels - 1)))
        k = int(np.round(z / n_pixels * (n_pixels - 1)))

        # Populate data at x,y,z
        Ex[step, i, j, k, 0] = ex
        Ey[step, i, j, k, 0] = ey
        Ez[step, i, j, k, 0] = ez

    np.save(PATH_TO_VOLUME_DATA+'Ex_3D_vol_' + str(n_pixels) + '_' + str(step) + '.npy', Ex)
    np.save(PATH_TO_VOLUME_DATA+'Ey_3D_vol_' + str(n_pixels) + '_' + str(step) + '.npy', Ey)
    np.save(PATH_TO_VOLUME_DATA+'Ez_3D_vol_' + str(n_pixels) + '_' + str(step) + '.npy', Ez)

# -----------------------------------------------
# Current | J

for step in range(n_timesteps):

    data = np.loadtxt(PATH_TO_VOLUME_DATA + J_VOLUME_DATA[:-5] + str(step) + '.txt', skiprows=3)

    Jx = np.zeros([2, n_pixels, n_pixels, n_pixels, 1])
    Jy = np.zeros([2, n_pixels, n_pixels, n_pixels, 1])
    Jz = np.zeros([2, n_pixels, n_pixels, n_pixels, 1])

    # Convert coordinates to indices and populate the Jx array
    for row in data:
        x, y, z, jx, jy, jz = row
        i = int(np.round(x / n_pixels * (n_pixels - 1)))
        j = int(np.round(y / n_pixels * (n_pixels - 1)))
        k = int(np.round(z / n_pixels * (n_pixels - 1)))

        # Populate data at x,y,z
        Jx[step, i, j, k, 0] = jx
        Jy[step, i, j, k, 0] = jy
        Jz[step, i, j, k, 0] = jz

    np.save(PATH_TO_VOLUME_DATA+'Jx_3D_vol_' + str(n_pixels) + '_' + str(step) + '.npy', Jx)
    np.save(PATH_TO_VOLUME_DATA+'Jy_3D_vol_' + str(n_pixels) + '_' + str(step) + '.npy', Jy)
    np.save(PATH_TO_VOLUME_DATA+'Jz_3D_vol_' + str(n_pixels) + '_' + str(step) + '.npy', Jz)

# TODO
# - Save max files which are used for normalization