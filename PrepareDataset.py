import os
from datetime import datetime
from os import walk, getcwd

import numpy as np
import re

from Exif import load_exif
import AutoNeuro
import Morphology


def load_dataset(file: str = 'newest'):
    if file == 'newest':
        for (_, _, filenames) in walk(getcwd() + '\\Training Data\\Datasets'):
            #print(dirpath, dirnames, filenames)
            filenames.sort()
            filename = filenames[-1]
            break
    else:
        filename = file
    path = getcwd() + '\\Training Data\\Datasets\\' + filename
    print(f'Loading dataset from {path}.')
    npz = np.load(path)
    x, y = npz['arr_0'], npz['arr_1']
    print(f'Loaded {x.shape[0]} training pictures in total')
    return x, y


def prepare_single_frame(photo, custom: dict = None):
    if custom is None:
        custom = Morphology.defaultOptions()
    do_continue = False
    try:
        res_autoneuro = AutoNeuro.face_landmarks_from_frame(photo)
        landmarks, ROIs = res_autoneuro["landmarks"], res_autoneuro["ROIs"]
        res_morpho = Morphology.analyze_eyes(ROIs, False, custom)
        # print("all good")
    except AssertionError:
        print(f"assertion error occured when analyzing frame {photo}, skipping")
        do_continue = True
    except Exception:
        print(f"other error occured when analyzing frame{photo}, skipping")
        do_continue = True

    if do_continue:
        return None
    flat = flatten_landmarks(landmarks, res_morpho)
    assert flat.shape == (1544,)
    return flat


def prepare_dataset(filename: str = None, must_include: str = None, exclude: str = None,
                    person: str = 'Jantek Mikulski', debug = False, custom: dict = None, print_errors = False):
    num_good = 0
    num_errs = 0
    print(f'Preparing the dataset ...')
    if filename is None:
        t = datetime.now()
        filename = t.strftime('%Y.%m.%d.%H.%M.%S')
    full_path = os.getcwd() + "\\Training Data\\Datasets\\" + filename + ".npz"
    dataset_x = []
    dataset_y = []
    file_paths = scope_files(person)
    if custom is None:
        custom = Morphology.defaultOptions()
    for photo in file_paths:
        metadata = load_exif(photo)
        if must_include:
            if not re.search(must_include, metadata['place']) and not re.search(must_include, metadata[
                'lightConditions']) and not re.search(must_include, metadata['info']):
                continue
        if exclude:
            if re.search(exclude, metadata['place']) or re.search(exclude, metadata['lightConditions']) or re.search(
                    exclude, metadata['info']):
                continue
        res_autoneuro = (AutoNeuro.face_landmarks_from_photo_batch([photo], with_ROI=True)[0])
        landmarks, ROIs, cursor, blink = res_autoneuro["landmarks"], res_autoneuro["ROIs"], res_autoneuro["cursor"], res_autoneuro["blink"]
        do_continue = False
        try:
            res_morpho = Morphology.analyze_eyes(ROIs, debug, custom)
            #print("all good")
        except AssertionError:
            if print_errors:
                print(f"assertion error occured when analyzing frame {photo}, skipping")
            do_continue = True
            num_errs += 1
        except Exception:
            if print_errors:
                print(f"other error occured when analyzing frame{photo}, skipping")
            do_continue = True
            num_errs += 1

        if do_continue:
            continue
        num_good += 1
        flat = flatten_landmarks(landmarks, res_morpho)
        assert flat.shape == (1544,)

        dataset_x.append(flat)
        cursor_and_blink = (*cursor, *blink)
        dataset_y.append(cursor_and_blink)
    dataset_x_arr = np.array(dataset_x)
    dataset_y_arr = np.array(dataset_y)
    np.savez(full_path, dataset_x_arr, dataset_y_arr)
    print(f"Finished preparing the dataset. Saved under {full_path}.")
    print(f"Got {num_errs} errors of total {num_errs+num_good} ({num_errs/(num_errs+num_good)*100:.2f}%).")


def flatten_landmarks(landmarks, res_morpho) -> np.ndarray:
    morpho_landmarks = []
    for idx, tpl in enumerate(res_morpho):
        if idx < 2:
            for idx2, dct in enumerate(tpl):
                if idx2 < 4:
                    stats = dct['stats']
                    stats = stats.reshape(-1).tolist()
                    for num in stats:
                        morpho_landmarks.append(float(num))

                    stats = dct['centroids']
                    stats = stats.reshape(-1).tolist()
                    for num in stats:
                        morpho_landmarks.append(float(num))
                else:
                    for dct2 in dct:
                        stats = dct2['coords']
                        for coord in stats:
                            morpho_landmarks.append(float(coord))

                        morpho_landmarks.append(dct2['size'])

        else:
            for num in tpl:
                morpho_landmarks.append(float(num))

    xyz = [(lm.x, lm.y, lm.z) for lm in landmarks]
    xyz = np.array(xyz).reshape(-1)
    return np.hstack((morpho_landmarks, xyz))


def scope_files(person: str):
    # photos = np.array([""], dtype=np.str_)
    photos = None
    for (dir_path, dir_names, file_names) in walk(getcwd() + "\\Training Data\\" + person):
        if file_names:
            paths = np.char.add(np.char.add(dir_path, "\\"), file_names)
            # photos = np.concatenate(photos, paths)
            # photos = photos + paths
            if photos is None:
                photos = paths.tolist()
            else:
                photos = photos + paths.tolist()
    return photos


if __name__ == "__main__":
    #prepare_dataset(must_include='3gen|gen3')
    x,y = load_dataset('newest')
    pass
