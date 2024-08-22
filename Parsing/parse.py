from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import struct
import os
from datetime import datetime
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.spatial import QhullError
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import imageio

# Constants and data types definitions
GDFNAMELEN = 16
GDFID = 94325877
t_dbl = int('0003', 16)
t_sval = 1024
t_arr = 2048
t_dir = 256
t_edir = 512

def gdftomemory(gdf_file):
    all = []

    def myprocfunc(params, data):
        all.append({'p': params, 'd': data})

    readgdf(gdf_file, myprocfunc)
    return all

def readgdf(gdf_file, procfunc):
    gdf_head = {}

    with open(gdf_file, 'rb') as f:
        gdf_id_check = struct.unpack('i', f.read(4))[0]
        if gdf_id_check != GDFID:
            raise RuntimeWarning('File is not a .gdf file')

        time_created = struct.unpack('i', f.read(4))[0]
        time_created = datetime.fromtimestamp(time_created)
        gdf_head['time_created'] = time_created.isoformat(' ')

        gdf_head['creator'] = f.read(GDFNAMELEN).decode().rstrip('\x00')
        gdf_head['destination'] = f.read(GDFNAMELEN).decode().rstrip('\x00')

        major = struct.unpack('B', f.read(1))[0]
        minor = struct.unpack('B', f.read(1))[0]
        gdf_head['gdf_version'] = f"{major}.{minor}"

        major = struct.unpack('B', f.read(1))[0]
        minor = struct.unpack('B', f.read(1))[0]
        gdf_head['creator_version'] = f"{major}.{minor}"

        major = struct.unpack('B', f.read(1))[0]
        minor = struct.unpack('B', f.read(1))[0]
        gdf_head['destination_version'] = f"{major}.{minor}"

        f.seek(2, 1)

        _parseblocks(f, {}, procfunc)

    return gdf_head

def _parseblocks(f, params, procfunc):
    data = {}
    while True:
        name = f.read(GDFNAMELEN)
        if len(name) == 0:
            procfunc(params, data)
            return
        name = name.decode().rstrip('\x00')
        type_, = struct.unpack('i', f.read(4))
        size, = struct.unpack('i', f.read(4))

        is_dir = (type_ & t_dir != 0)
        is_edir = (type_ & t_edir != 0)
        is_sval = (type_ & t_sval != 0)
        is_arr = (type_ & t_arr != 0)

        dattype = type_ & 255

        if is_sval:
            if dattype == t_dbl:
                value = struct.unpack('d', f.read(8))[0]
            elif dattype == 0x0010:  # t_null
                pass
            elif dattype == 0x0001:  # t_ascii
                value = f.read(size).decode().rstrip('\x00')
            elif dattype == 0x0002:  # t_s32
                value = struct.unpack('i', f.read(4))[0]
            else:
                value = f.read(size)
        elif is_arr:
            if dattype == t_dbl:
                value = np.frombuffer(f.read(size), dtype=np.double)
                data[name] = value
            else:
                value = f.read(size)

        if is_dir:
            myparams = params.copy()
            myparams[name] = value
            _parseblocks(f, myparams, procfunc)
        elif is_edir:
            procfunc(params, data)
            return
        elif is_sval:
            params[name] = value



def global_xyz_real(data):
    z_max_range = 0
    x_min = float("inf")
    x_max = float("-inf")
    y_min = float("inf")
    y_max = float("-inf")
    z_min = float("inf")
    z_max = float("-inf")

    for step in range(len(data) - 1):
        x = np.array(data[step].get("d").get("x"))
        y = np.array(data[step].get("d").get("y"))
        z = np.array(data[step].get("d").get("z"))
        q = np.array(data[step].get("d").get("q"))

        mask = q != 0
        x_real = x[mask]
        y_real = y[mask]
        z_real = z[mask]

        x_min = min(x_min, x_real.min())
        x_max = max(x_max, x_real.max())
        y_min = min(y_min, y_real.min())
        y_max = max(y_max, y_real.max())
        z_min = min(z_min, z_real.min())
        z_max = max(z_max, z_real.max())

        z_range = z_real.max() - z_real.min()
        z_max_range = max(z_max_range, z_range)

    print('xmin: ', x_min)
    print('xmax: ', x_max)
    print('ymin: ', y_min)
    print('ymax: ', y_max)
    print('zmin: ', z_min)
    print('zmax: ', z_max)
    print('z_max_range: ', z_max_range)

    return x_min, x_max, y_min, y_max, z_min, z_max, z_max_range


def process_step(args):
    step, data, xbins, ybins, z_max_range, pixel_dimensions, bin_volume, OUTPUT_PATH = args
    c = 2.99792458e8

    # Extract coordinates and charge information
    x = np.array(data[step].get("d").get("x"))
    y = np.array(data[step].get("d").get("y"))
    z = np.array(data[step].get("d").get("z"))
    q = np.array(data[step].get("d").get("q"))

    # Mask for real and dummy particles
    real_mask = q != 0
    dummy_mask = q == 0

    components = {
        "Ex": np.array(data[step].get("d").get("fEx")),
        "Ey": np.array(data[step].get("d").get("fEy")),
        "Ez": np.array(data[step].get("d").get("fEz")),
        "Bx": np.array(data[step].get("d").get("fBx")),
        "By": np.array(data[step].get("d").get("fBy")),
        "Bz": np.array(data[step].get("d").get("fBz")),
        "q": np.array(data[step].get("d").get("q")),
    }

    # Betas
    velocities = {
        "Bx": np.array(data[step].get("d").get("Bx")),
        "By": np.array(data[step].get("d").get("By")),
        "Bz": np.array(data[step].get("d").get("Bz")),
    }

    binning_result = {}

    zbins = np.linspace(z.min(), z.min() + z_max_range, pixel_dimensions[2] + 1)

    for key, data_comp in components.items():
        # If binning rho, add values and divide by bin volume
        if key == 'q':
            hist, _ = np.histogramdd((x[real_mask], y[real_mask], z[real_mask]), bins=[xbins, ybins, zbins], weights=data_comp[real_mask], density=True)
            binning_result[key] = hist
        else:
            points = np.array([x[dummy_mask], y[dummy_mask], z[dummy_mask]]).T

            # Create a mesh grid for interpolation
            grid_x, grid_y, grid_z = np.meshgrid(
                0.5 * (xbins[:-1] + xbins[1:]),
                0.5 * (ybins[:-1] + ybins[1:]),
                0.5 * (zbins[:-1] + zbins[1:]),
                indexing='ij'
            )
            grid_points = np.array([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]).T

            try:
                # Use LinearNDInterpolator for interpolation
                interpolator = LinearNDInterpolator(points, data_comp[dummy_mask])
                interpolated = interpolator(grid_points)

                # Handle potential NaN values from interpolation (due to out-of-bounds points)
                interpolated[np.isnan(interpolated)] = 0

                hist = interpolated.reshape(pixel_dimensions)
            except QhullError as e:
                print(f"Qhull error encountered: {e}")
                hist = np.zeros(pixel_dimensions)  # As a fallback, return zeros

            binning_result[key] = hist

    for component, binned in binning_result.items():
        np.save(OUTPUT_PATH + f'{component}_3D_vol_{pixel_dimensions[0]}_{pixel_dimensions[1]}_{pixel_dimensions[2]}_{step}.npy', binned)

    Jx = components["q"][real_mask] * velocities["Bx"][real_mask] * c / bin_volume
    Jy = components["q"][real_mask] * velocities["By"][real_mask] * c / bin_volume
    Jz = components["q"][real_mask] * velocities["Bz"][real_mask] * c / bin_volume

    j_components = {
        "Jx": Jx,
        "Jy": Jy,
        "Jz": Jz,
    }

    for j_key, j_data in j_components.items():
        hist, _ = np.histogramdd((x[real_mask], y[real_mask], z[real_mask]), bins=[xbins, ybins, zbins], weights=j_data, density=True)
        np.save(OUTPUT_PATH + f'{j_key}_3D_vol_{pixel_dimensions[0]}_{pixel_dimensions[1]}_{pixel_dimensions[2]}_{step}.npy', hist)

    B_max_local = max(components["Bx"][real_mask].max(), components["By"][real_mask].max(), components["Bz"][real_mask].max())
    J_max_local = max(Jx.max(), Jy.max(), Jz.max())

    return B_max_local, J_max_local


def process_data(data, pixel_dimensions, OUTPUT_PATH):

    B_max_global = float("-inf")
    J_max_global = float("-inf")

    x_min, x_max, y_min, y_max, z_min, z_max, z_max_range = global_xyz_real(data)

    xbins = np.linspace(x_min, x_max, pixel_dimensions[0] + 1)
    ybins = np.linspace(y_min, y_max, pixel_dimensions[1] + 1)

    bin_volume = z_max_range * (xbins[1] - xbins[0]) * (ybins[1] - ybins[0])

    pool_args = [(step, data, xbins, ybins, z_max_range, pixel_dimensions, bin_volume, OUTPUT_PATH) for step in range(len(data) - 1)]

    with Pool(cpu_count()) as pool:
        results = pool.map(process_step, pool_args)

    for B_max, J_max in results:
        B_max_global = max(B_max_global, B_max)
        J_max_global = max(J_max_global, J_max)

    np.save(OUTPUT_PATH + 'Bxyz_max.npy', B_max_global)
    np.save(OUTPUT_PATH + 'J_max_max_all_128.npy', J_max_global)

    print("Fixed bin volume:", bin_volume)
    print("B_max_global:", B_max_global)
    print("J_max_global:", J_max_global)



def compute_max_abs_value(variable_name, data_length, pixel_dimensions, OUTPUT_PATH):
    max_abs_value = 0
    for i in range(data_length):
        file_path = OUTPUT_PATH + f'{variable_name}_3D_vol_{pixel_dimensions[0]}_{pixel_dimensions[1]}_{pixel_dimensions[2]}_{i}.npy'
        binned_data = np.load(file_path)
        max_abs_value = max(max_abs_value, np.abs(binned_data).max())
    return max_abs_value



def plot_frame(args):
    i, data, xbins, ybins, pixel_dimensions, variable_name, OUTPUT_PATH, max_abs_value = args
    x_centers = 0.5 * (xbins[:-1] + xbins[1:])
    y_centers = 0.5 * (ybins[:-1] + ybins[1:])

    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(121, projection='3d')

    file_path = OUTPUT_PATH + f'{variable_name}_3D_vol_{pixel_dimensions[0]}_{pixel_dimensions[1]}_{pixel_dimensions[2]}_{i}.npy'
    binned_data = np.load(file_path)

    if np.all(binned_data == 0):
        print(f"File contains only zeros: {file_path}")
        return None

    # MIGHT NEED A REAL PARTICLE MASK HERE
    z_data = data[i].get("d").get("z")
    z = np.array(z_data)
    z_min_current = z.min()
    z_max_current = z.max()

    zbins = np.linspace(z_min_current, z_max_current, pixel_dimensions[2] + 1)
    z_centers = 0.5 * (zbins[:-1] + zbins[1:])

    X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')

    x_flattened = X.flatten()
    y_flattened = Y.flatten()
    z_flattened = Z.flatten()

    non_zero_indices = np.nonzero(binned_data.flatten())

    if len(non_zero_indices[0]) > 0:
        sc = ax.scatter(z_flattened[non_zero_indices], x_flattened[non_zero_indices],
                        y_flattened[non_zero_indices], c=binned_data.flatten()[non_zero_indices],
                        cmap='viridis', s=50, vmin=-max_abs_value, vmax=max_abs_value)
        fig.colorbar(sc, ax=ax, label=variable_name)
    else:
        ax.text2D(0.5, 0.5, "No Data", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    ax.set_xlim(z_min_current, z_max_current)
    ax.set_ylim(xbins[0], xbins[-1])
    ax.set_zlim(ybins[0], ybins[-1])

    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.set_title(f'Binned 3D Scatter Plot of {variable_name} - Timestep {i}')

    ax.view_init(elev=30, azim=-60)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5))

    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
    ax.zaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))

    ax2 = fig.add_subplot(122)
    z_slice_index = pixel_dimensions[2] // 2
    slice_data = binned_data[:, :, z_slice_index]
    extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
    im = ax2.imshow(slice_data.T, extent=extent, origin='lower', aspect='equal', cmap='viridis', vmin=-max_abs_value, vmax=max_abs_value)
    fig.colorbar(im, ax=ax2, label=f'{variable_name} at Z={z_centers[z_slice_index]:.2f}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'2D Slice of {variable_name} at Z={z_centers[z_slice_index]:.2f}')

    filename = f'{variable_name}_timestep_{i}.png'
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved frame {i} as {filename}")
    return filename



def plot_variable(variable_name, data_length, xbins, ybins, data, pixel_dimensions, OUTPUT_PATH):

    max_abs_value = compute_max_abs_value(variable_name, data_length, pixel_dimensions, OUTPUT_PATH)
    print(f'Max absolute value for {variable_name}: {max_abs_value}')

    pool_args = [(i, data, xbins, ybins, pixel_dimensions, variable_name, OUTPUT_PATH, max_abs_value) for i in range(data_length)]

    # Plot in parallel
    with Pool(cpu_count()) as pool:
        filenames = pool.map(plot_frame, pool_args)

    # Compile the gif
    filenames = [fname for fname in filenames if fname is not None]
    gif_filename = f'{variable_name}_output.gif'
    with imageio.get_writer(gif_filename, mode='I', duration=0.1, loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            print(f"Appended {filename} to GIF")

    print(f'GIF saved as {gif_filename}')

    # Delete temporary png files
    for filename in filenames:
        os.remove(filename)

    print(f'Removed {len(filenames)} temporary PNG files.')



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dims', help='Specify output dimensions: <x>,<y>,<z>', type=str)
    parser.add_argument('--dir_out', help='Specify the output directory for parsing and plotting: /path/to/output/', type=str)
    parser.add_argument('--plot', action='store_true', help='Plot the data after processing')
    parser.add_argument('--parse', action='store_true', help='Parse the data')
    args = parser.parse_args()

    # Default configuration
    OUTPUT_PATH = '/sdf/scratch/rfar/jcurcio/Volume_Data/'
    data = gdftomemory("PINN_trainingData_08.gdf")
    pixel_dimensions = (128, 128, 128)

    if args.dims:
        pixel_dimensions = tuple([int(dim) for dim in args.dims.split(',')])
        print('Output dimensions: ', pixel_dimensions)

    if args.dir_out:
        OUTPUT_PATH = args.dir_out
        print('Output path: ', OUTPUT_PATH)

    if args.parse:
        process_data(data, pixel_dimensions, OUTPUT_PATH)
        print('npy files saved.')

    if args.plot:

        global_x_min, global_x_max, global_y_min, global_y_max, _, _, _ = global_xyz_real(data)

        data_length = len(data) - 1
        xbins = np.linspace(global_x_min, global_x_max, pixel_dimensions[0] + 1)
        ybins = np.linspace(global_y_min, global_y_max, pixel_dimensions[1] + 1)

        plot_variable('Ex', data_length, xbins, ybins, data, pixel_dimensions, OUTPUT_PATH)

        print('Plotting complete.')

