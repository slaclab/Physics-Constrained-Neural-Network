import tensorflow as tf

def check_gpu():
    # Check if TensorFlow can detect GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs detected: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu}")
    else:
        print("No GPUs detected. Please check your TensorFlow installation and environment.")

def test_gpu_usage():
    # Create a simple operation and force it to run on GPU if available
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        # Create a random tensor and perform a simple computation
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        c = tf.matmul(a, b)

        # Print the result
        print("Result of matrix multiplication:", c.numpy())

    # Check which device was used for the operation
    print("Device used for matrix multiplication:")
    for op in c.op.inputs:
        print(op.device)

if __name__ == "__main__":
    # Check GPU availability
    check_gpu()

    # Test if the computation runs on GPU
    test_gpu_usage()
