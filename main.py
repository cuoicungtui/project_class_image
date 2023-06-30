import os
import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import cv2


# Check ảnh ko hợp lệ khi vào model
path = '../tu_data_image/'
err_image = []

for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        x = os.path.join(dirname, filename)
        try:
            image_data = tf.io.read_file(x)
            # Giải mã hình ảnh PNG với RGBA
            image = tf.image.decode_png(image_data, channels=4)  # Chế độ RGBA
        except tf.errors.InvalidArgumentError as e:
            err_image.append(filename)


class ImageDataset:
    def __init__(self, directory, exclude_files, batch_size=32, image_size=(256, 256)):
        self.directory = directory
        self.exclude_files = exclude_files
        self.batch_size = batch_size
        self.image_size = image_size

    def _generator(self):
#         file_list = os.listdir(self.directory)
        # săp xêp theo tên
        file_list = sorted(os.listdir(self.directory)) 
        for file in file_list:
            if file not in self.exclude_files:
                file_path = os.path.join(self.directory, file)
                try:
                    image_data = tf.io.read_file(file_path)
                    image = tf.io.decode_png(image_data, channels=4)  # Chế độ RGBA
                    yield image
                except tf.errors.InvalidArgumentError as e:
                    print(file_path)
    
    def create_dataset(self):
        dataset = tf.data.Dataset.from_generator(
            self._generator,
            output_signature=tf.TensorSpec(shape=(None, None, 4), dtype=tf.uint8)
        )
        dataset = dataset.map(lambda x: tf.image.resize(x, self.image_size))
        dataset = dataset.batch(self.batch_size)
        return dataset

# path data input
path = '../tu_data_image/'
exclude_files = err_image
batch_size = 32
image_size = (256, 256)

dataset = ImageDataset(path, exclude_files, batch_size, image_size)
dataset = dataset.create_dataset()

model_path = './model_save/model1.h5'
model_loader = load_model(model_path)

predicted_labels = []
for images in dataset:
#     print(images.shape)
    predictions = model_loader.predict(images)
    predicted_labels.extend(predictions.argmax(axis=1))

file_dic = sorted(os.listdir(path)) 

while_file = []
black_file = []
index =0
for i in file_dic:
    if i not in exclude_files:
        if index < len(predicted_labels):
            if predicted_labels[index] == 1:
                while_file.append(i)
            else:
                black_file.append(i)
            index += 1
        else:
            print("Error: Index out of range for predicted_labels list.")

def save_list_to_text(file_path, my_list):
    with open(file_path, 'w') as file:
        for item in my_list:
            file.write(str(item) + '\n')   

file_path = 'White.txt'
save_list_to_text(file_path, while_file)
file_path = 'Black.txt'
save_list_to_text(file_path, black_file) 
