import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import cv2
import argparse
import shutil
from tqdm import tqdm

list_image_err = []

# Check ảnh ko hợp lệ khi vào model
# path = '../tu_data_image/'
# err_image = []

# for dirname, _, filenames in os.walk(path):
#     for filename in filenames:
#         x = os.path.join(dirname, filename)
#         try:
#             image_data = tf.io.read_file(x)
#             # Giải mã hình ảnh PNG với RGBA
#             image = tf.image.decode_png(image_data, channels=4)  # Chế độ RGBA
#         except tf.errors.InvalidArgumentError as e:
#             err_image.append(filename)

class ImageDataset:
    def __init__(self, directory, batch_size=32, image_size=(256, 256)):
        self.directory = directory
        self.batch_size = batch_size
        self.image_size = image_size

    def _generator(self):
#         file_list = os.listdir(self.directory)
        # săp xêp theo tên
        file_list = sorted(os.listdir(self.directory)) 
        for file in file_list:
            file_path = os.path.join(self.directory, file)
            try:
                image_data = tf.io.read_file(file_path)
                image = tf.io.decode_png(image_data, channels=4)  # Chế độ RGBA
                yield image
            except tf.errors.InvalidArgumentError as e:
                list_image_err.append(file)
                # print(file_path)
    
    def create_dataset(self):
        dataset = tf.data.Dataset.from_generator(
            self._generator,
            output_signature=tf.TensorSpec(shape=(None, None, 4), dtype=tf.uint8)
        )
        dataset = dataset.map(lambda x: tf.image.resize(x, self.image_size))
        dataset = dataset.batch(self.batch_size)
        return dataset

def copy_files_with_matching_names(source_directory, destination_directory, file_list_path):
    # Kiểm tra nếu thư mục đích không tồn tại thì tạo mới
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    with open(file_list_path, 'r') as file:
        for line in file:
            file_name = line.strip()
            source_file_path = os.path.join(source_directory, file_name)
            destination_file_path = os.path.join(destination_directory, file_name)
            if os.path.exists(source_file_path):
                shutil.copy(source_file_path, destination_file_path)

def copy_files_with_list(source_directory, destination_directory, file_list):
    # Kiểm tra nếu thư mục đích không tồn tại thì tạo mới
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    for line in tqdm(file_list):
        file_name = line.strip()
        source_file_path = os.path.join(source_directory, file_name)
        destination_file_path = os.path.join(destination_directory, file_name)
        if os.path.exists(source_file_path):
            shutil.copy(source_file_path, destination_file_path)

# def save_list_to_text(file_path, my_list):
#     with open(file_path, 'w') as file:
#         for item in my_list:
#             file.write(str(item) + '\n')   

# file_path = 'White.txt'
# save_list_to_text(file_path, while_file)
# file_path = 'Black.txt'
# save_list_to_text(file_path, black_file) 

def main():
    parser = argparse.ArgumentParser(description='image classification')
    parser.add_argument('--path_dict', type=str, default=None, help='path to dict image png')
    parser.add_argument('--path_save_white', type=str, default=None, help='path to save white')
    parser.add_argument('--path_save_black', type=str, default=None, help='path to save black')
    parser.add_argument('--path_save_err', type=str, default=None, help='path to save err')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size to predict type int')
    args = parser.parse_args()
    path_dict_image_png = args.path_dict
    path_save_white = args.path_save_white
    path_save_black = args.path_save_black
    path_save_err = args.path_save_err
    BATCH_SIZE = args.batch_size
    print(path_dict_image_png)
    print(path_save_white)
    print(path_save_black)
    print(path_save_err)
    print('BATCH_SIZE : ',BATCH_SIZE)
    print("Input Done!")

    # load model
    model_path = './model_save/model1.h5'
    model_loader = load_model(model_path)

    #ceate dataset
    batch_size =BATCH_SIZE
    image_size = (256, 256)
    dataset = ImageDataset(path_dict_image_png, batch_size, image_size)
    dataset = dataset.create_dataset()
    predicted_labels = []
    print("Start predict image")
    for images in dataset:
        predictions = model_loader.predict(images)
        predicted_labels.extend(predictions.argmax(axis=1))

    file_dic = sorted(os.listdir(path_dict_image_png)) 
    while_file = []
    black_file = []
    index = 0
    for i in file_dic:
        if i not in list_image_err:
            if index < len(predicted_labels):
                if predicted_labels[index] == 1:
                    while_file.append(i)
                else:
                    black_file.append(i)
                index += 1
    print("Done predict image")
    print("white: ", len(while_file), "black: ", len(black_file), "err: ", len(list_image_err))

    #copy file split white and black and err image to folder
    print("Start copy file")
    print("Copy white image")
    copy_files_with_list(path_dict_image_png, path_save_white, while_file)
    print("Copy black image")
    copy_files_with_list(path_dict_image_png, path_save_black, black_file)
    print("Copy err image")
    copy_files_with_list(path_dict_image_png, path_save_black, list_image_err)
    print("Done copy file")

if __name__ == '__main__':
    main()