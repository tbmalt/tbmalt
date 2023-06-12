# -*- coding: utf-8 -*-
"""Elemental reference data.

Reference data pertaining to chemical elements & their properties are located
here. As the `chemical_symbols` & `atomic_numbers` attributes are frequently
used they have been made accessible from the `tbmalt.data` namespace.

Attributes:
    chemical_symbols (List[str]): List of chemical symbols whose indices are
        the atomic numbers of the associated elements; i.e.
        `chemical_symbols[6]` will yield `"C"`.
    atomic_numbers (Dict[str, int]): Dictionary keyed by chemical symbols &
        valued by atomic numbers. This is used to get the atomic number
        associated with a given chemical symbol.
    gamma_cutoff (Dict[tuple, Tensor]): Dictionary keyed by pairs of atomic
        numbers & valued by pre-calculated cutoff distances for short range
        part of gamma calculations.
    gamma_element_list: (List[int]): List of atomic numbers of the elements whose
        cutoff distances for gamma calculations are available from
        pre-calculations.

"""
from typing import List, Dict
import torch
Tensor = torch.Tensor


# Highest atomic number the project can deal with. Currently this is only used
# by the `OrbitalInfo` class.
MAX_ATOMIC_NUMBER = 120

# Chemical symbols of the elements. Neutronium is included to ensure the index
# matches the atomic number and to assist with batching behaviour.
chemical_symbols: List[str] = [
    # Period zero
    'X' ,
    # Period one
    'H' , 'He',
    # Period two
    'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne',
    # Period three
    'Na', 'Mg', 'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar',
    # Period four
    'K',  'Ca', 'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # Period five
    'Rb', 'Sr', 'Y' , 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I' , 'Xe',
    # Period six
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt',
    'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    # Period seven
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U' , 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
    'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]

# Dictionary for looking up an element's atomic number.
atomic_numbers: Dict[str, int] = {sym: z for z, sym in
                                  enumerate(chemical_symbols)}

# Dictionary of pre-calculated cutoff values of atom pairs for shortgamma
# calculations. These are used exclusively by the `gamma_exponential_pbc`
# function; which constructs the gamma term via the exponential method when
# operating upon periodic systems.
gamma_cutoff: Dict[tuple, Tensor] = {
    (1, 1, 'cutoff'): torch.tensor([20.024999999999999]),
    (1, 6, 'cutoff'): torch.tensor([22.037500000000001]),
    (1, 7, 'cutoff'): torch.tensor([19.521874999999998]),
    (1, 8, 'cutoff'): torch.tensor([18.515625000000000]),
    (1, 16, 'cutoff'): torch.tensor([23.043750000000003]),
    (1, 79, 'cutoff'): torch.tensor([30.087500000000002]),
    (6, 1, 'cutoff'): torch.tensor([22.037500000000001]),
    (6, 6, 'cutoff'): torch.tensor([22.540625000000002]),
    (6, 7, 'cutoff'): torch.tensor([22.037500000000001]),
    (6, 8, 'cutoff'): torch.tensor([20.528124999999999]),
    (6, 14, 'cutoff'): torch.tensor([30.087500000000002]),
    (6, 16, 'cutoff'): torch.tensor([24.050000000000001]),
    (6, 79, 'cutoff'): torch.tensor([29.081250000000004]),
    (7, 1, 'cutoff'): torch.tensor([19.521874999999998]),
    (7, 6, 'cutoff'): torch.tensor([22.037500000000001]),
    (7, 7, 'cutoff'): torch.tensor([20.024999999999999]),
    (7, 8, 'cutoff'): torch.tensor([19.018749999999997]),
    (8, 1, 'cutoff'): torch.tensor([18.515625000000000]),
    (8, 6, 'cutoff'): torch.tensor([20.528124999999999]),
    (8, 7, 'cutoff'): torch.tensor([19.018749999999997]),
    (8, 8, 'cutoff'): torch.tensor([17.006250000000001]),
    (8, 16, 'cutoff'): torch.tensor([23.043750000000003]),
    (8, 79, 'cutoff'): torch.tensor([28.075000000000003]),
    (14, 14, 'cutoff'): torch.tensor([33.003124999999997]),
    (14, 6, 'cutoff'): torch.tensor([30.087500000000002]),
    (16, 1, 'cutoff'): torch.tensor([23.043750000000003]),
    (16, 6, 'cutoff'): torch.tensor([24.050000000000001]),
    (16, 8, 'cutoff'): torch.tensor([23.043750000000003]),
    (16, 16, 'cutoff'): torch.tensor([25.056249999999999]),
    (16, 79, 'cutoff'): torch.tensor([30.087500000000002]),
    (79, 1, 'cutoff'): torch.tensor([30.087500000000002]),
    (79, 6, 'cutoff'): torch.tensor([29.081250000000004]),
    (79, 8, 'cutoff'): torch.tensor([28.075000000000003]),
    (79, 16, 'cutoff'): torch.tensor([30.087500000000002]),
    (79, 79, 'cutoff'): torch.tensor([32.100000000000001]),
    (0, 0, 'cutoff'): torch.tensor([1.1]), (0, 1, 'cutoff'): torch.tensor([1.1]),
    (0, 6, 'cutoff'): torch.tensor([1.1]), (0, 7, 'cutoff'): torch.tensor([1.1]),
    (0, 8, 'cutoff'): torch.tensor([1.1]), (0, 14, 'cutoff'): torch.tensor([1.1]),
    (0, 16, 'cutoff'): torch.tensor([1.1]), (0, 79, 'cutoff'): torch.tensor([1.1]),
    (1, 0, 'cutoff'): torch.tensor([1.1]), (6, 0, 'cutoff'): torch.tensor([1.1]),
    (7, 0, 'cutoff'): torch.tensor([1.1]), (8, 0, 'cutoff'): torch.tensor([1.1]),
    (14, 0, 'cutoff'): torch.tensor([1.1]), (16, 0, 'cutoff'): torch.tensor([1.1]),
    (79, 0, 'cutoff'): torch.tensor([1.1])}

# List of atomic numbers of the elements with pre-calculated cutoff distances
# for shortgamma calculations.
gamma_element_list: List[int] = [0, 1, 6, 8, 16, 79]
