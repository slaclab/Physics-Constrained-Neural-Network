import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator, ScalarFormatter

# Constants
PATH_TO_VOLUME_DATA = '/sdf/scratch/rfar/jcurcio/Volume_Data/'
n_pixels = 128
pixel_dimensions = (128, 128, 128)
n_timesteps = 126
test_start_idx = 65  # Starting index for the test data (25% of the data after 75% training)

# Load the trained model
model_A = load_model('../Trained_Models/PCNN.h5')

# Function to load data for a specific timestep
def load_test_data(timestep):
    Jx = np.load(PATH_TO_VOLUME_DATA + f'Jx_3D_vol_{pixel_dimensions[0]}_{pixel_dimensions[1]}_{pixel_dimensions[2]}_{timestep}.npy').reshape([1, n_pixels, n_pixels, n_pixels, 1])
    Jy = np.load(PATH_TO_VOLUME_DATA + f'Jy_3D_vol_{pixel_dimensions[0]}_{pixel_dimensions[1]}_{pixel_dimensions[2]}_{timestep}.npy').reshape([1, n_pixels, n_pixels, n_pixels, 1])
    Jz = np.load(PATH_TO_VOLUME_DATA + f'Jz_3D_vol_{pixel_dimensions[0]}_{pixel_dimensions[1]}_{pixel_dimensions[2]}_{timestep}.npy').reshape([1, n_pixels, n_pixels, n_pixels, 1])
    Bx = np.load(PATH_TO_VOLUME_DATA + f'Bx_3D_vol_{pixel_dimensions[0]}_{pixel_dimensions[1]}_{pixel_dimensions[2]}_{timestep}.npy')
    By = np.load(PATH_TO_VOLUME_DATA + f'By_3D_vol_{pixel_dimensions[0]}_{pixel_dimensions[1]}_{pixel_dimensions[2]}_{timestep}.npy')
    Bz = np.load(PATH_TO_VOLUME_DATA + f'Bz_3D_vol_{pixel_dimensions[0]}_{pixel_dimensions[1]}_{pixel_dimensions[2]}_{timestep}.npy')
    return Jx, Jy, Jz, Bx, By, Bz

# Iterate over the test data timesteps
for timestep in range(test_start_idx, n_timesteps):
    Jx, Jy, Jz, Bx_original, By_original, Bz_original = load_test_data(timestep)

    # Latent space inputs, un-used, all zeros
    z_input = np.zeros([1, 8, 8, 8, 1]).astype(np.float32)

    # Predict Ax, Ay, Az
    Ax_predicted, _ = model_A.predict([Jx, z_input])
    Ay_predicted, _ = model_A.predict([Jy, z_input])
    Az_predicted, _ = model_A.predict([Jz, z_input])

    # Reshape the predictions to remove the single channel dimension
    Ax_predicted = Ax_predicted[0, ..., 0]  # Shape: (128, 128, 128)
    Ay_predicted = Ay_predicted[0, ..., 0]  # Shape: (128, 128, 128)
    Az_predicted = Az_predicted[0, ..., 0]  # Shape: (128, 128, 128)

    # Compute numerical derivatives to get the magnetic fields
    def compute_derivatives(A, dx, dy, dz):
        A_y = np.gradient(A, axis=1) / dy
        A_z = np.gradient(A, axis=2) / dz
        A_x = np.gradient(A, axis=0) / dx
        return A_x, A_y, A_z

    dx = dy = dz = 1  # Assuming unit spacing for simplicity; adjust if necessary

    Ax1_x, Ax1_y, Ax1_z = compute_derivatives(Ax_predicted, dx, dy, dz)
    Ay1_x, Ay1_y, Ay1_z = compute_derivatives(Ay_predicted, dx, dy, dz)
    Az1_x, Az1_y, Az1_z = compute_derivatives(Az_predicted, dx, dy, dz)

    # Magnetic Fields
    Bx_predicted = Az1_y - Ay1_z  # Bx = ∂Az/∂y - ∂Ay/∂z
    By_predicted = Ax1_z - Az1_x  # By = ∂Ax/∂z - ∂Az/∂x
    Bz_predicted = Ay1_x - Ax1_y  # Bz = ∂Ay/∂x - ∂Ax/∂y

    print(f'Timestep {timestep}:')
    print('Number of zero elements in Bx:', len(Bx_predicted.flatten()) - np.count_nonzero(Bx_predicted))
    print('Number of zero elements in By:', len(By_predicted.flatten()) - np.count_nonzero(By_predicted))
    print('Number of zero elements in Bz:', len(Bz_predicted.flatten()) - np.count_nonzero(Bz_predicted))

    # PLOTTING
    # ------------------------------------------------------------------------------------------------------------

    # Generate coordinate arrays
    x = np.linspace(0, 1, n_pixels)
    y = np.linspace(0, 1, n_pixels)
    z = np.linspace(0, 1, n_pixels)

    # Create meshgrid for coordinates
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 3D PLOTTING

    # Get the min and max values for the colormap
    Bx_min = Bx_predicted.min()
    Bx_max = Bx_predicted.max()

    # Plot the 3D scatter plot
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(121, projection='3d')

    sc = ax.scatter(Z.flatten(), X.flatten(), Y.flatten(), c=Bx_predicted.flatten(), cmap='viridis', s=1, vmin=Bx_min, vmax=Bx_max)
    fig.colorbar(sc, ax=ax, label='Bx')

    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.set_title(f'3D Scatter Plot of Predicted Magnetic Field Bx at Timestep {timestep}')

    # 2D SLICE PLOTTING

    # Slice the data for a specific z-coordinate
    z_slice_index = n_pixels // 2
    x_slice = X[:, :, z_slice_index]
    y_slice = Y[:, :, z_slice_index]
    Bx_slice = Bx_predicted[:, :, z_slice_index]

    ax2 = fig.add_subplot(122)
    sc2 = ax2.imshow(Bx_slice.T, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', aspect='equal', cmap='viridis', vmin=Bx_min, vmax=Bx_max)
    fig.colorbar(sc2, ax=ax2, label='Bx')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'2D Slice of Bx at Z = {z[z_slice_index]:.2f} at Timestep {timestep}')

    plt.show()
    plt.savefig(f'Predicted_3D_2D_timestep_{timestep}.png')
    plt.close(fig)

