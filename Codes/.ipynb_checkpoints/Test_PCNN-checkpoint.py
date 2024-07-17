import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the trained model
loaded_model = tf.keras.models.load_model('../Trained_Models/PCNN.h5', compile=False)

step = 10
n_pixels = 128
PATH_TO_VOLUME_DATA = '/pscratch/sd/j/jcurcio/pcnn/Volume_Data/'

loaded_model.summary()

# Load the component for the next timestep (step + 1)
X_in_next = np.load(f'{PATH_TO_VOLUME_DATA}/q_3D_vol_{n_pixels}_{step + 1}.npy').reshape([1, n_pixels, n_pixels, n_pixels, 1])

# Create a dummy ES_in for the next timestep, if you have actual data, load it accordingly
ES_in_next = np.zeros([1, 8, 8, 8, 1]).astype(np.float32)

# Make a prediction
prediction = loaded_model.predict([X_in_next, ES_in_next])

# Print the outputs
A_output, adapt_in_output = prediction
print("Prediction for timestep", step + 1, ":")
print("A output shape:", A_output.shape)
print(A_output)
print("Adapt in output shape:", adapt_in_output.shape)
print(adapt_in_output)

# PLOTTING CODE
# Reshape A_output to match voxel dimensions
A_output = A_output.reshape((n_pixels, n_pixels, n_pixels))

# Create a figure with 3 subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})

# Extract slices for each axis
x_slice = A_output[n_pixels // 2, :, :]
y_slice = A_output[:, n_pixels // 2, :]
z_slice = A_output[:, :, n_pixels // 2]

# Function to plot non-zero elements of a slice
def plot_non_zero(ax, data, title, xlabel, ylabel):
    non_zero_indices = data != 0
    x, y = np.nonzero(non_zero_indices)
    z = data[non_zero_indices]
    ax.scatter(x, y, z, c=z, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

# Plot x_slice
plot_non_zero(axs[0], x_slice, f'X Slice at index {n_pixels // 2}', 'Y', 'Z')

# Plot y_slice
plot_non_zero(axs[1], y_slice, f'Y Slice at index {n_pixels // 2}', 'X', 'Z')

# Plot z_slice
plot_non_zero(axs[2], z_slice, f'Z Slice at index {n_pixels // 2}', 'X', 'Y')

# Display the plot
plt.show()
