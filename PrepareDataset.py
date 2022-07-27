import os
from datetime import datetime
from os import walk, getcwd

import numpy as np


def prepare_dataset(filename: str = None, must_include: str = None, exclude: str = None,
                    person: str = "Jantek Mikulski"):
    if filename is None:
        t = datetime.now()
        filename = t.strftime('%Y.%m.%d.%H.%M.%S')
    full_path = "Datasets\\" + filename + ".npz"

    file_paths = scope_files(person)



def scope_files(person: str):
    photos = np.array([], dtype=np.str_)
    for (dir_path, dir_names, file_names) in walk(getcwd() + "\\Training Data\\" + person):
        if file_names:
            paths = np.char.add(dir_path, file_names)
            photos = np.concatenate(photos, paths)
    return photos


prepare_dataset()
