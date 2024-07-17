import numpy as np
import os

def create_dummy_data(folder, num_files):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i in range(num_files):
        # Create a dummy array of shape (1, 384, 320)
        dummy_array = np.random.rand(1, 384, 320).astype(np.float32)
        # Save the array as a .npy file
        file_path = os.path.join(folder, f'dummy_{i}.npy')
        np.save(file_path, dummy_array)
    print(f'Created {num_files} dummy files in {folder}')

# Define the folders and number of files to create
train_folder = 'data_train'
val_folder = 'data_val'
num_train_files = 100
num_val_files = 20

# Create dummy data for training and validation
create_dummy_data(train_folder, num_train_files)
create_dummy_data(val_folder, num_val_files)
