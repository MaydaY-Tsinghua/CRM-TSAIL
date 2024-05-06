import zipfile
import os
directory = "/root/CRM/GSO"
# Get a list of all files and directories in the specified directory
files_and_directories = os.listdir(directory)

# If you want only files, you can filter out directories like this:
files = [f for f in files_and_directories if os.path.isfile(os.path.join(directory, f))]
# print(files)
output_dir = "/root/CRM/GSO_extracted"
for i, file in enumerate(files):
    if file.endswith(".zip"):
        with zipfile.ZipFile(f"{directory}/{file}", 'r') as zip_ref:
            zip_ref.extractall(f"{output_dir}/{i}")
            print(f"Extracted {file}")
        # os.remove(f"{directory}/{file}")
        # print(f"Removed {file}")