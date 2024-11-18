import os
import tarfile

# Define paths
dataset_folder = 'compressed_datasets'
data_folder = 'data'

try: 
    os.mkdir(data_folder)
    os.mkdir(data_folder + '/Kifu')
    os.mkdir(data_folder + '/Top50')
except FileExistsError:
    print('Datasets are already decompressed.')
    exit()

# Define the mapping of tgz files to their respective folders
folder_mapping = {
    'gokif2.tgz': 'Kifu',
    'gokif3.tgz': 'Kifu',
    'go9-large.tgz': 'Kifu',
    'go9.tgz': 'Top50',
}

# Create the destination folders if they don't exist
for folder in set(folder_mapping.values()):
    path = os.path.join(data_folder, folder)
    os.makedirs(path, exist_ok=True)
    os.mkdir(f'{path}/nine')
    os.mkdir(f'{path}/nine/nine_weak')
    os.mkdir(f'{path}/nine/nine_strong')
    os.mkdir(f'{path}/nine/nine_mixed')
    os.mkdir(f'{path}/nine/nine_not_needed')
    os.mkdir(f'{path}/other')
    os.mkdir(f'{path}/too_small')
    os.mkdir(f'{path}/error')


for tgz_file, target_folder in folder_mapping.items():
    tgz_path = os.path.join(dataset_folder, tgz_file)
    extract_path = os.path.join(data_folder, target_folder)
    
    with tarfile.open(tgz_path, 'r:gz') as tar_ref:
        for member in tar_ref.getmembers():
            # Remove any initial directory component from the file path
            member.name = os.path.basename(member.name)
            tar_ref.extract(member, extract_path)
        
        print(f"{tgz_file} has been decompressed to {extract_path}")

print("Datasets have been decompressed and placed in their respective folders in /data.")