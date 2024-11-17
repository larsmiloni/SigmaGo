import os
import glob
import numpy as np
import math
import pickle
import time
from typing import List
from pysgf import SGF
from cnn_channels import get_states_new

# Define paths
cwd = os.getcwd()
path = os.path.join(cwd, 'data')
pickle_root = 'pickles/pickleFile_'

kifu_path = os.path.join(path, 'Kifu/')
os.makedirs('pickles', exist_ok=True)

# Parameters
HANDICAP_GAMES = 0
BATCH_SIZE = 1000  # Adjust based on your memory capacity
GAMES_PER_FILE = 1500000  # Number of games per pickle file

def move_files_to_subdirectories(path_name):
    """Organize SGF files into subdirectories based on size and presence of handicap."""
    print("Organizing files...")
    count= 0

    for file in glob.glob(os.path.join(path_name, "*.sgf")):
        count += 1
        print(f"Files processed: {count}")
        if (count==1000):
            break
        try:
            parsed_file = SGF.parse_file(file)
            if parsed_file.get_property('HA'):
                # Skip handicap games
                os.rename(file, os.path.join(path_name, "handicap", os.path.basename(file)))
                continue
            match_size = parsed_file.get_property('SZ')
            if match_size == '9':
                os.rename(file, os.path.join(path_name, "nine", os.path.basename(file)))
            else:
                os.rename(file, os.path.join(path_name, "other", os.path.basename(file)))
        except Exception as e:
            print(f"Error processing {file}: {e}")
            os.rename(file, os.path.join(path_name, "error", os.path.basename(file)))
        
        

    

def parse_and_pickle_batches(path_name, batch_size=BATCH_SIZE, games_per_file=GAMES_PER_FILE):
    """Process SGF files in batches and pickle the results."""
    sgf_files = glob.glob(os.path.join(path_name, "nine", "nine_strong", "*.sgf"))
    total_files = len(sgf_files)
    print(f"Total files to process: {total_files}")

    features = []
    labels = []
    pickle_index = 1
    game_count = 0

    start_time = time.time()

    for idx, sgf_file in enumerate(sgf_files):
        try:
            print(f"Processing file {idx + 1}/{total_files}: {sgf_file}")
            get_states_new(sgf_file, features, labels)
            game_count += 1

            if game_count % batch_size == 0 or idx == total_files - 1:
                # Convert to numpy arrays
                features_array = np.array(features, dtype='float32')
                labels_array = np.array(labels, dtype='float32')

                # Shuffle data
                permutation = np.random.permutation(len(features_array))
                features_array = features_array[permutation]
                labels_array = labels_array[permutation]

                # Pickle the batch
                pickle_file = f"{pickle_root}{pickle_index}.pickle"
                with open(pickle_file, 'wb') as f:
                    pickle.dump({'dataset': features_array, 'labels': labels_array}, f, pickle.HIGHEST_PROTOCOL)
                print(f"Saved pickle file: {pickle_file}")

                # Reset for next batch
                features.clear()
                labels.clear()
                pickle_index += 1

                # Optional: Break into smaller pickles based on GAMES_PER_FILE
                if game_count >= games_per_file:
                    game_count = 0  # Reset game count for new pickle file

                # Print timing information
                elapsed_time = time.time() - start_time
                print(f"Processed {idx + 1} files in {elapsed_time:.2f} seconds.")
                start_time = time.time()

        except Exception as e:
            print(f"Error processing file {sgf_file}: {e}")
            os.rename(sgf_file, os.path.join(path_name, "error", os.path.basename(sgf_file)))

    print("All files processed.")

# Main execution
if __name__ == "__main__":
    # Ensure necessary directories exist
    for subdir in ['nine', 'other', 'error', 'handicap']:
        os.makedirs(os.path.join(kifu_path, subdir), exist_ok=True)

    # Move files into subdirectories
    move_files_to_subdirectories(kifu_path)

    # Process files in batches and pickle the data
    parse_and_pickle_batches(kifu_path, batch_size=BATCH_SIZE)