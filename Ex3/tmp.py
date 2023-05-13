import os

# Specify the directory path
directory_path = "data/midi_files/"
# Traverse through all the files in the directory and its subdirectories
for root, dirs, files in os.walk(directory_path):
    for file in files:
        # Get the current file name and convert it to lowercase
        current_file = os.path.join(root, file)
        new_file = current_file.lower()

        # Rename the file to its lowercase version
        os.rename(current_file, new_file)