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
    npz = np.load(getcwd() + '\\Training Data\\Datasets\\' + filename)
    x, y = npz['arr_0'], npz['arr_1']
    return x, y


def prepare_dataset(filename: str = None, must_include: str = None, exclude: str = None,
                    person: str = 'Jantek Mikulski'):
    if filename is None:
        t = datetime.now()
        filename = t.strftime('%Y.%m.%d.%H.%M.%S')
    full_path = os.getcwd() + "\\Training Data\\Datasets\\" + filename + ".npz"
    dataset_x = []
    dataset_y = []
    file_paths = scope_files(person)
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
        landmarks, ROIs, cursor = res_autoneuro["landmarks"], res_autoneuro["ROIs"], res_autoneuro["cursor"]
        do_continue = False
        try:
            res_morpho = Morphology.analyze_eyes(ROIs)
            print("all good")
        except AssertionError:
            print(f"assertion error occured when analyzing frame {photo}, skipping")
            do_continue = True
        except Exception:
            print(f"other error occured when analyzing frame{photo}, skipping")
            do_continue = True

        if do_continue:
            continue
        flat = flatten_landmarks(landmarks, res_morpho)
        assert flat.shape == (1544,)

        dataset_x.append(flat)
        dataset_y.append(cursor)
    dataset_x_arr = np.array(dataset_x)
    dataset_y_arr = np.array(dataset_y)
    np.savez(full_path, dataset_x_arr, dataset_y_arr)


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
    #prepare_dataset(filename="test pose", must_include="test pose")
    load_dataset('test pose.npz.npz')
    pass
