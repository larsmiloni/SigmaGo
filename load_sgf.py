from typing import List
import numpy as np
import glob
import re
import pickle
import timeit
import os
from pysgf import SGF
import math
from cnn_channels import get_states

cwd = os.getcwd()
path = cwd + '/data'
pickleRoot = 'pickles/pickleFile_'

kifuPath = path + '/Kifu/'
top50Path = path + '/Top50/'

os.mkdir('pickles')
os.mkdir('pickles_mixed')
os.mkdir('checkpoints')
os.mkdir('trainResults')

HANDICAP_GAMES = 0


def randomize(dataset, labels):
    permutation = np.random.permutation(len(dataset))
    shuffled_dataset = dataset[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def transform_data(function_name, dataset, labelset, dataset_file="dataset_memmap.npy", labelset_file="labelset_memmap.npy"):
    if len(dataset) > 0:
        # Determine the shape of the expanded arrays
        total_rows = 2 * len(dataset)
        
        # Create memory-mapped arrays on disk
        expanded_dataset = np.memmap(dataset_file, dtype='float32', mode='w+', shape=(total_rows, 83))
        expanded_labelset = np.memmap(labelset_file, dtype='float32', mode='w+', shape=(total_rows, 83))

        for x in range(len(dataset)):
            # Copy original data to the first half
            expanded_dataset[x, :] = dataset[x]
            expanded_labelset[x, :] = labelset[x]

            # Transform and store in the second half
            temp_data = dataset[x, :81].reshape(9, 9)
            expanded_dataset[len(dataset) + x, :81] = function_name(temp_data).reshape(81)
            expanded_dataset[len(dataset) + x, 81:] = dataset[x, 81:]

            temp_label = labelset[x, :81].reshape(9, 9)
            expanded_labelset[len(labelset) + x, :81] = function_name(temp_label).reshape(81)
            expanded_labelset[len(labelset) + x, 81:] = labelset[x, 81:]

        # Flush changes to disk
        expanded_dataset.flush()
        expanded_labelset.flush()

        return expanded_dataset, expanded_labelset

def pickleFiles(features, labels):

    games = len(features)
    games_per_file = 1_500_000
    pickleFileCount = math.ceil(games / games_per_file)
    print("Number of pickle files needed for this sub dataset = ", pickleFileCount)
    for i in range(pickleFileCount):
        try:
            startRange = i * games_per_file
            if i == pickleFileCount - 1:
                stopRange = startRange + games % games_per_file
            else:
                stopRange = startRange + games_per_file
            print("pickling file: ", str(i + 1))
            print("startRange: ", startRange, " stopRange: ", stopRange)
            pickle_file = pickleRoot + str(i + 1) + '.pickle'
            f = open(pickle_file, 'wb')
            save = {
                'dataset': features[startRange:stopRange],
                'labels': labels[startRange:stopRange],

            }
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            f.close()

        except Exception as e:
            print('Unable to save data to pickle file: ', e)
            raise


def countFiles(pathName):
    count = 0
    for filename in os.listdir(pathName):
        path = os.path.join(pathName, filename)

        if os.path.isfile(path):
            count += 1
    return count


def parseMoves(file_name: str, labels: List, features: List):
    get_states(file_name, features, labels)


def loadFiles(pathName, rankingSystem):
    sgfFiles = glob.glob(pathName + "*.sgf")

    for file in sgfFiles:

        if file.endswith(".sgf"):
            filename = os.path.basename(file)
            filesize = os.path.getsize(file)
            if filesize < 180:
                os.rename(file, pathName + "too_small/" + filename)
                continue
            try:
                parsed_file = SGF.parse_file(file) 
                match_size = parsed_file.get_property('SZ')

                if parsed_file.get_property('HA'):
                    """Skip handicap games"""
                    HANDICAP_GAMES += 1
                    continue

                if match_size == '9':
                    os.rename(file, pathName + "nine/" + filename)
                else:
                    os.rename(file, pathName + "other/" + filename)
            except Exception as e:
                os.rename(file, pathName + "error/" + filename)
                print('Error in figuring out board size: ', file, e)

    print("File count for loading files from data path: ", pathName)
    print("9x9: ", countFiles(pathName + "nine/"))
    print("other: ", countFiles(pathName + "other/"))
    print("Error count after sorting by size: ",
          countFiles(pathName + "error/"))

    if (rankingSystem == "ELO" or rankingSystem == "Traditional"):
        # Don't do this for the pro games
        nine_files = glob.glob(pathName + "nine/" + "*.sgf")
        for file in nine_files:
            if file.endswith(".sgf"):
                filename = os.path.basename(file)
                try:
                    parsed_data = SGF.parse_file(file)
                    pattern = r"\((\d+)\)"
                    w_rank_text = parsed_data.get_property('PB')
                    b_rank_text = parsed_data.get_property('PW')

                    if rankingSystem == 'ELO':
                        # Select games where players are 1 kyu or better (2000 ELO ranking)
                        b_rank = int(re.search(pattern, b_rank_text).group(1))
                        w_rank = int(re.search(pattern, w_rank_text).group(1))

                        if b_rank >= 2000 and w_rank >= 2000:
                            os.rename(file, pathName +
                                      "nine/nine_strong/" + filename)
                        else:
                            os.rename(file,  pathName +
                                      "nine/nine_weak/" + filename)
                    else:
                        # Ranking is KYU / DAN / PRO system
                        b_kyu_dan = re.search(
                            r"BR\[([0-9]*)([kdpKDP])\]*", b_rank_text).group(2)

                        w_kyu_dan = re.search(
                            r"WR\[([0-9]*)([kdpKDP])\]*", w_rank_text).group(2)

                        sufficient_ranks = {'d', 'D', 'p', 'P'}

                        if w_kyu_dan in sufficient_ranks and b_kyu_dan in sufficient_ranks:
                            os.rename(file, pathName +
                                      "nine/nine_strong/" + filename)
                        else:
                            os.rename(file,  pathName +
                                      "nine/nine_weak/" + filename)

                except Exception as e:
                    print('Error processing player rankings: ', file, " ",  e)
                    os.rename(file, pathName + "error/" + filename)

        print("Weak 9x9 games found: ", countFiles(pathName + "nine/nine_weak"))
        print("Strong 9x9 games found: ",  countFiles(
            pathName + "nine/nine_strong"))
        print("Error count after determining rank: ",
              countFiles(pathName + "error/"))
    else:
        nine_files = glob.glob(pathName + "nine/" + "*.sgf")

        for file in nine_files:

            if file.endswith(".sgf"):
                filename = os.path.basename(file)
                os.rename(file, pathName + "nine/nine_strong/" + filename)
        print("Strong 9x9 games found: ",  countFiles(
            pathName + "nine/nine_strong"))
    # End of ranking system checker

    labels = []
    features = []

    i = 0
    start_time = timeit.default_timer()

    nine_files_strong = glob.glob(
        pathName + "nine/" + "nine_strong/" + "*.sgf")

    k = 0
    for nfile in nine_files_strong:
        k = k + 1
        if k == 100:
            break
        if nfile.endswith(".sgf"):
            filename = os.path.basename(nfile)
        else:
            continue

        if (i % 500 == 0):
            print("files done: ",  i, " time taken: ",
                  int(timeit.default_timer() - start_time))
            start_time = timeit.default_timer()
        i = i + 1

        try:
            parseMoves(nfile, labels, features)
        except Exception as e:
            print('Error in reading moves in SGF file: ', nfile, e)
            os.rename(nfile, pathName + "error/" + filename)

    labels_array = np.array(labels).astype('float32')
    features_array = np.array(features).astype('float32')

    print("Type of labels:", type(labels_array))
    print("Type of features:", type(features_array))
    print("Shape of labels:", labels_array.shape)
    print("Shape of features:", features_array.shape)

    #features_array, labels_array = transform_data(
    #    np.fliplr, features_array, labels_array)
    #print("fliplr done")
    #print("labels shape: ", labels_array.shape)
    #print("features shape: ", features_array.shape)

    #features_array, labels_array = transform_data(
    #    np.flipud, features_array, labels_array)
    #print("flipud done")
    #print("labels shape: ", labels_array.shape)
    #print("features shape: ", features_array.shape)

    #features_array, labels_array = transform_data(
    #    np.rot90, features_array, labels_array)
    #print("rot90 done")
    #print("labels shape: ", labels_array.shape)
    #print("features shape: ", features_array.shape)

    #print("before randomize")
    #features_array, labels_array = randomize(features_array, labels_array)
    #print("after randomize")

    pickleFiles(features_array, labels_array)

print("Handicap games found:", HANDICAP_GAMES)
loadFiles(top50Path, "ELO")