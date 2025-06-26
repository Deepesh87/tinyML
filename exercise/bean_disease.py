import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import os
import zipfile
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil

train_file_name= "train.zip"
validation_file_name = "validation.zip"
test_file_name = "test.zip"
files_to_read= [train_file_name, validation_file_name, test_file_name]


def delete_directory(path):
    """
    Deletes the directory at the specified path, including all subfolders and files.

    Parameters:
    path (str): The path of the directory to delete.
    """
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
        print(f"Deleted directory: {path}")
    else:
        print(f"Directory does not exist: {path}")


# function to copy the dataset to a destination directory
def copy_dataset(source, destination, files_to_read):
    if not os.path.exists(destination):
        os.makedirs(destination)
    for item in files_to_read:
        source_path = os.path.join(source, item)
        destination_path = os.path.join(destination, item)
        if os.path.isdir(source_path): # Check if it's a directory
            # If it's a directory, copy the entire directory
            copy_dataset(source_path, destination_path)
            print(f"Copied directory: {source_path} to {destination_path}")
        else: # If it's a file, copy the file
            tf.io.gfile.copy(source_path, destination_path)
            print(f"Copied file: {source_path} to {destination_path}")


source= "C:/Users/xperi/Downloads/bean disease"
destination= "C:/Users/xperi/OneDrive/Desktop/study/datasets/bean disease" 
delete_directory(destination)  # Clean up the destination directory before copying
copy_dataset(source, destination, files_to_read)