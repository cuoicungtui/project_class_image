import os
import shutil

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

# Example usage

source_directory = 'D:/tu_CNN/tu_data_image'
destination_directory = './data/White'
file_list_path = './White.txt'
copy_files_with_matching_names(source_directory, destination_directory, file_list_path)

source_directory = 'D:/tu_CNN/tu_data_image'
destination_directory = './data/Black'
file_list_path = './black.txt'
copy_files_with_matching_names(source_directory, destination_directory, file_list_path)