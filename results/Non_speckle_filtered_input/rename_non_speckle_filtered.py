import os

def rename_files(directory_path, new_name_suffix):
    for filename in os.listdir(directory_path):
        old_file_path = os.path.join(directory_path, filename)
        if os.path.isfile(old_file_path):
            # Split the file name and extension
            name, extension = os.path.splitext(filename)
            arr = name.split("_")


            # Create the new name with the specified suffix
            new_name = arr[0] + new_name_suffix + extension

            # Create the new file path
            new_file_path = os.path.join(directory_path, new_name)

            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f'Renamed {filename} to {new_name}')

# Example usage:
directory_path = '/home/vidhu/Downloads/FSEG_py-master/results/Non_speckle_filtered_input'  # Replace this with the path to your target directory
new_name_suffix = '_non_speckle_filtered'  # Replace this with the suffix you want to add to the filenames

rename_files(directory_path, new_name_suffix)

