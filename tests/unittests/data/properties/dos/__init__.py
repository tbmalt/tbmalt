"""This acts as a data loader for the dos/CH4 test data."""
from numpy import loadtxt
import torch
from glob import glob
from re import search
from os.path import basename, join, dirname
"""
Notes:
    Think about placing on a custom device.
    Look at abstracting numpy to torch conversion
"""


def _load_bases(path='./'):
    """Load orbs information."""
    file = join(path, 'bases.csv')
    keys = open(file, 'r').readline()[2:-1].split(', ')
    values = torch.tensor(loadtxt(file, delimiter=','), dtype=int)
    return {k: v for k, v in zip(keys, values.T)}


def _load_eigenvalues(path='./'):
    """Load eigenvalue vector."""
    return torch.tensor(loadtxt(join(path, 'eigenvalues.csv'), delimiter=','))


def _load_eigenvectors(path='./'):
    """Load coefficient matrix, i.e. the eigenvectors."""
    return torch.tensor(loadtxt(join(path, 'eigenvectors.csv'), delimiter=','))


def _load_overlap(path='./'):
    """Load overlap matrix."""
    return torch.tensor(loadtxt(join(path, 'overlap.csv'), delimiter=','))


def _load_fermi(path='./'):
    """Load Fermi energy."""
    return loadtxt(join(path, 'fermi.csv'), delimiter=',')[()]


def _load_sigma(path='./'):
    """Load sigma smearing value."""
    return loadtxt(join(path, 'sigma.csv'), delimiter=',')[()]


def _load_dos(path='./'):
    """Load density of states."""
    file = join(path, 'dos.csv')
    keys = open(file, 'r').readline()[2:-1].split(', ')
    values = torch.tensor(loadtxt(file, delimiter=','))
    return {k: v for k, v in zip(keys, values.T)}


def _load_pdos(path='./'):
    """Load projected density of states."""
    data = {}
    for f in glob(join(path, 'pdos_*.csv')):
        element = search(r'(?<=pdos_).*(?=\.csv)', basename(f)).group(0)
        keys = open(f, 'r').readline()[2:-1].split(', ')
        values = torch.tensor(loadtxt(f, delimiter=','))
        data[element] = {k: v for k, v in zip(keys, values.T)}
    return data


def _load_data(path='./'):
    data = {
        'bases': _load_bases(path),
        'eigenvalues': _load_eigenvalues(path),
        'eigenvectors': _load_eigenvectors(path),
        'overlap': _load_overlap(path),
        'fermi': _load_fermi(path),
        'sigma': _load_sigma(path),
        'dos': _load_dos(path),
        'pdos': _load_pdos(path),
    }

    return data

if __name__ != '__main__':
    path = dirname(__file__)
    H2 = _load_data(f'{path}/H2')
    CH4 = _load_data(f'{path}/CH4')
    HCOOH = _load_data(f'{path}/HCOOH')
