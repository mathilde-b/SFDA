#!/usr/bin/env python3.6
import argparse
from argparse import Namespace
import warnings
from sys import argv
from typing import Dict, Iterable
from pathlib import Path
from functools import partial
import nibabel as nib
import numpy as np
from skimage.io import imread, imsave
from utils import read_anyformat_image,read_nii_image
from lib import write_nii

def create(filename: str):
    print(filename)
    filename_ = filename.split('.nii')[0]
    acc: np.ndarray = read_nii_image(filename)
    for i in range(0,acc.shape[2]):
        sl = acc[:,:,i]
        sl = sl[ np.newaxis,...]
        print(sl.shape)
        write_nii(sl,filename_+'_'+str(i))

def main():

    folder = Path(argv[1])
    targets: Iterable[str] = map(str, folder.glob("*.nii.gz"))
    for target in targets:
        create(target)


if __name__ == "__main__":
    main()
