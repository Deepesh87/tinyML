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


# function to copy a directory from a source to to a destination
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

# extract training dataset
local_zip = os.path.join(destination, train_file_name)
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall(destination + '/train')

#extract validation dataset
local_zip = os.path.join(destination, validation_file_name)
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall(destination + '/validation')
zip_ref.close()

#extract test dataset
local_zip = os.path.join(destination, validation_file_name)
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall(destination + '/test')
zip_ref.close()

# delete the zip files after extraction
for file in files_to_read:
    file_path = os.path.join(destination, file)
    print(f"Deleting file: {file_path}")
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    else:
        print(f"File does not exist: {file_path}")


train_datagen = ImageDataGenerator(
      rescale=1./224,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest'
      )

validation_datagen = ImageDataGenerator(
      rescale=1./224,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest'
)



TRAIN_DIRECTORY_LOCATION = os.path.join(destination, 'train/train')
VAL_DIRECTORY_LOCATION = os.path.join(destination, 'validation/validation')
TARGET_SIZE = (224, 224)
# 3 classes: healthy, bean_rust, bean_bacterial_blight
CLASS_MODE = "categorical"  # Use 'categorical' for multi-class classification

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIRECTORY_LOCATION,
    target_size = TARGET_SIZE,  
    batch_size = 128,
    class_mode = CLASS_MODE,
)

validation_generator = validation_datagen.flow_from_directory(
    VAL_DIRECTORY_LOCATION,
    target_size = TARGET_SIZE,  
    batch_size = 128,
    class_mode = CLASS_MODE,
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2), 
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax') # Use 'softmax' for multi-class classification
])

# This will print a summary of your model when you're done!
model.summary()

LOSS_FUNCTION = 'categorical_crossentropy'
OPTIMIZER = RMSprop(learning_rate=0.0001)

model.compile(
    loss = LOSS_FUNCTION,
    optimizer = OPTIMIZER,
    metrics = ['accuracy']
)

NUM_EPOCHS = 20 #YOUR CODE HERE#

history = model.fit(
      train_generator, 
      epochs = NUM_EPOCHS,
      verbose = 1,
      validation_data = validation_generator)

# summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.xlim([0,NUM_EPOCHS])
plt.ylim([0.4,1.0])
plt.show()

