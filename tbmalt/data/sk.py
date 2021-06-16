# -*- coding: utf-8 -*-
"""Slater-Koster (SK) reference data.

Reference data of frequently used when performing SK transformation or read
SK tables.

Attributes:
    SkIntType (Literal[str]): Type of SK data.
    int_s (List[tuple]): List of interactions when the maximum of quantum ℓ
        number is 0.
    int_p (List[tuple]): List of interactions when the maximum of quantum ℓ
        number is 1.
    int_d (List[tuple]): List of interactions when the maximum of quantum ℓ
        number is 2.
    int_f (List[tuple]): List of interactions when the maximum of quantum ℓ
        number is 3.
    hdf_suffix (list): All the allowed suffixes of binary hdf files.
    skf_suffix (list): All the suffixes of SKF files.

"""
from typing import List


# The interactions: ddσ, ddπ, ddδ, ...
int_s: List[tuple] = [(0, 0, 0)]

int_p: List[tuple] = [(1, 1, 0), (1, 1, 1), (0, 1, 0), (0, 0, 0)]

int_d: List[tuple] = [
    (2, 2, 0), (2, 2, 1), (2, 2, 2), (1, 2, 0), (1, 2, 1),
    (1, 1, 0), (1, 1, 1), (0, 2, 0), (0, 1, 0), (0, 0, 0)]

int_f: List[tuple] = [
    (3, 3, 0), (3, 3, 1), (3, 3, 2), (3, 3, 3), (2, 3, 0), (2, 3, 1),
    (2, 3, 2), (2, 2, 0), (2, 2, 1), (2, 2, 2), (1, 3, 0), (1, 3, 1),
    (1, 2, 0), (1, 2, 1), (1, 1, 0), (1, 1, 1), (0, 3, 0), (0, 2, 0),
    (0, 1, 0), (0, 0, 0)]

hdf_suffix = ['HD', 'Hd', 'hd', 'HDF', 'Hdf', 'hdf', 'HDF5', 'Hdf5', 'hdf5',
              'H5', 'h5']
skf_suffix = ['SKF', 'Skf', 'skf']
