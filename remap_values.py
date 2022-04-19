#!/usr/bin/env python3.6

import warnings
from sys import argv
from typing import Dict, Iterable
from pathlib import Path
from functools import partial
import os
import numpy as np
from skimage.io import imread, imsave
from utils import mmap_, read_anyformat_image
import nibabel as nib



def remap(changes: Dict[int, int], filename: str):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        img = nib.load(filename)
        save_folder = str(Path(filename).parent)+'New/'
        base = os.path.basename(filename)
        os.makedirs(save_folder,exist_ok=True)
        acc = read_anyformat_image(filename)
        #img = nib.load(filename)
        #acc = img.get_data()
        print(filename)
        print(np.unique(acc))
        assert set(np.unique(acc)).issubset(changes), (set(changes), np.unique(acc))

        for a, b in changes.items():
            acc[acc == a] = b

        acc = nib.Nifti1Image(acc,img.affine,img.header)
        nib.save(acc,save_folder+base)

def main():

    folder = Path(argv[1])
    changes = eval(argv[2])
    targets: Iterable[str] = map(str, folder.glob("*.nii"))
    for target in targets:
        remap(changes,target)

if __name__ == "__main__":
    main()
