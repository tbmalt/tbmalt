#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 16:10:30 2021

@author: gz_fan
"""
from typing import Union, Optional, Literal
import torch
from dscribe.descriptors import ACSF
from ase.build import molecule
from torch import Tensor
from tbmalt.ml.acsf import Acsf
from tbmalt import Geometry, Basis
from tbmalt.data.elements import chemical_symbols
from tbmalt.common.batch import pack
from tbmalt.data.units import length_units

# Set some global parameters which only used here
torch.set_default_dtype(torch.float64)
ch4 = molecule('CH4')
nh3 = molecule('NH3')
h2o = molecule('H2O')
h2o2 = molecule('H2O2')
h2 = molecule('H2')
cho = molecule('CH3CHO')
shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}
_electronnegativity = {1: torch.tensor([2.2]), 6: torch.tensor([2.55]),
                       7: torch.tensor([3.04]), 8: torch.tensor([3.44])}


class AcsfTest(Acsf):
    """Test Acsf class."""

    def __init__(self,
                 geometry: object, basis: object, shell_dict: dict,
                 g1_params: Union[float, Tensor],
                 g2_params: Optional[Tensor] = None,
                 g3_params: Optional[Tensor] = None,
                 g4_params: Optional[Tensor] = None,
                 g5_params: Optional[Tensor] = None,
                 unit: Literal['bohr', 'angstrom'] = 'bohr',
                 element_resolve: Optional[bool] = True,
                 atom_like: Optional[bool] = False):
        super().__init__(geometry, basis, shell_dict,
                         g1_params, unit, element_resolve, atom_like)

    def __call__(self,
                 g1_params: Optional[Tensor] = None,
                 g2_params: Optional[Tensor] = None,
                 g3_params: Optional[Tensor] = None,
                 g4_params: Optional[Tensor] = None,
                 g5_params: Optional[Tensor] = None):

        if g1_params is not None:
            self.fc, self.g1 = self.g2(g1_params)
        _g = self.g1.clone()
        if g2_params is not None:
            _g, self.g2 = self.g2(_g, g2_params)
        if g4_params is not None:
            _g, self.g4 = self.g4(_g, g4_params)
        if g5_params is not None:
            _g, self.g5 = self.g5(_g, g5_params)


        # if atom_like is True, return g in batch with atom_like, else return
        # g, which the first dimension equals to the size of all atoms in batch
        if self.atom_like:
            self.g = torch.zeros(*self.atomic_numbers.shape, _g.shape[-1])
            self.g[self.atomic_numbers.ne(0)] = _g
        else:
            self.g = _g

    def g4(self, g, g4_params):
        """Test function for G5 parameters."""
        d_vect = self.geometry.distance_vectors / length_units['angstrom']
        d_vect = d_vect.unsqueeze(0) if d_vect.dim() == 3 else d_vect
        batch = d_vect.shape[0]
        dist = self.distances / length_units['angstrom']

        fc = self.fc
        _g4 = torch.zeros(batch, *dist.squeeze().shape, dist.squeeze().shape[-1])
        d_vect = d_vect.squeeze()
        dist = dist.squeeze()
        fc = self.fc.squeeze()
        _g4 = torch.zeros(*dist.squeeze().shape, dist.squeeze().shape[-1])
        _cos = torch.zeros(*dist.squeeze().shape, dist.squeeze().shape[-1])
        _exp = torch.zeros(*dist.squeeze().shape, dist.squeeze().shape[-1])
        __fc = torch.zeros(*dist.squeeze().shape, dist.squeeze().shape[-1])
        _dist = torch.zeros(*dist.squeeze().shape, dist.squeeze().shape[-1])
        # lamb, eta, zeta = g4_params
        eta, zeta, lamb = g4_params
        for i in range(d_vect.shape[0]):
            for j in range(d_vect.shape[1]):
                for k in range(d_vect.shape[1]):
                    if i != j and i != k:
                        cos = (d_vect[i, j] * d_vect[i, k]).sum(-1) / \
                            dist[i, j] / dist[i, k]
                        exp = torch.exp(-eta * (
                            dist[i, j] ** 2 + dist[i, k] ** 2 + dist[j, k] ** 2))
                        _fc = fc[i, j] * fc[i, k] * fc[j, k]
                        _cos[i, j, k] = cos
                        _exp[i, j, k] = exp
                        __fc[i, j, k] = _fc
                        _dist[i, j, k] = (d_vect[i, j] * d_vect[i, k]).sum(-1)
                        _g4[i, j, k] = 2 ** (1 - zeta) * (
                            1 + lamb * cos) ** zeta * exp * _fc
        # ijk and ikj will be the same angle
        _g4 = (_g4 * 0.5).sum(-1).sum(-1).unsqueeze(1)

        return (torch.cat([g, _g4], dim=1), _g4) if self.element_resolve else \
            (torch.cat([self.g, _g4.sum(-1)], dim=1), _g4)

    def g5(self, g, g5_params):
        """Test function for G5 parameters."""
        d_vect = self.geometry.distance_vectors / length_units['angstrom']
        d_vect = d_vect.unsqueeze(0) if d_vect.dim() == 3 else d_vect
        batch = d_vect.shape[0]
        dist = self.distances / length_units['angstrom']

        fc = self.fc
        _g5 = torch.zeros(batch, *dist.squeeze().shape, dist.squeeze().shape[-1])
        d_vect = d_vect.squeeze()
        dist = dist.squeeze()
        fc = self.fc.squeeze()
        _g5 = torch.zeros(*dist.squeeze().shape, dist.squeeze().shape[-1])
        lamb, eta, zeta = g5_params

        for i in range(d_vect.shape[0]):
            for j in range(d_vect.shape[1]):
                for k in range(d_vect.shape[1]):
                    if i != j and i != k:
                        cos = (d_vect[i, j] * d_vect[i, k]).sum(-1) / \
                            dist[i, j] / dist[i, k]
                        exp = torch.exp(-eta * (
                            dist[i, j] ** 2 + dist[i, k] ** 2))
                        _fc = fc[i, j] * fc[i, k]
                        _g5[i, j, k] = 2 ** (1 - zeta) * (
                            1 + lamb * cos) ** zeta * exp * _fc

        # ijk and ikj will be the same angle
        _g5 = (_g5 * 0.5).sum(-1).sum(-1).unsqueeze(1)

        return (torch.cat([g, _g5], dim=1), _g5) if self.element_resolve else \
            (torch.cat([self.g, _g5.sum(-1)], dim=1), _g5)


def test_single_g1(device):
    """Test G1 values in single geometry."""
    rcut = 6.0

    # 1. Molecule test
    geo = Geometry.from_ase_atoms(ch4)
    basis = Basis(geo.atomic_numbers, shell_dict)
    species = geo.chemical_symbols
    acsf = Acsf(geo, basis, shell_dict, g1_params=rcut,
                element_resolve=True)
    acsf()

    # Get reference
    acsf_t = ACSF(species=species, rcut=rcut)
    acsf_t_g = torch.from_numpy(acsf_t.create(ch4))

    assert torch.max(abs(acsf_t_g - acsf.g)) < 1E-6, 'tolerance check'

    acsf_sum = Acsf(geo, basis, shell_dict, g1_params=rcut,
                    element_resolve=False)
    acsf_sum()
    assert torch.max(abs(acsf_t_g.sum(-1) - acsf_sum.g)) < 1E-6

    # 2. Periodic system test
    ch4.cell = [1, 3, 3]
    geo = Geometry.from_ase_atoms(ch4)
    basis = Basis(geo.atomic_numbers, shell_dict)
    species = geo.chemical_symbols
    acsfp = Acsf(geo, basis, shell_dict, g1_params=rcut,
                 element_resolve=True)
    acsfp()

    # Get reference
    acsf_t = ACSF(species=species, rcut=rcut, periodic=True)
    acsf_t_gp = torch.from_numpy(acsf_t.create(ch4))


def test_batch_g1(device):
    """Test G1 values in batch geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms([ch4, h2o])
    basis = Basis(geo.atomic_numbers, shell_dict)
    species = [chemical_symbols[ii] for ii in geo.unique_atomic_numbers()]
    acsf = Acsf(geo, basis, shell_dict, g1_params=rcut)
    acsf()

    # get reference
    acsf_d = ACSF(species=species, rcut=rcut)
    acsf_d_g1 = acsf_d.create([ch4, h2o])

    assert torch.max(abs(torch.from_numpy(acsf_d_g1[0]) -
                         acsf.g[: acsf_d_g1[0].shape[0]])) < 1E-6
    assert torch.max(abs(torch.from_numpy(acsf_d_g1[1]) -
                         acsf.g[acsf_d_g1[0].shape[0]:])) < 1E-6

    acsf_sum = Acsf(geo, basis, shell_dict, g1_params=rcut,
                    element_resolve=False)
    acsf_sum()
    assert torch.max(abs(torch.cat([
        torch.from_numpy(ii) for ii in acsf_d_g1]).sum(-1) - acsf_sum.g)) < 1E-6


def test_single_g2(device):
    """Test G2 values in single geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms(ch4)
    basis = Basis(geo.atomic_numbers, shell_dict)
    species = geo.chemical_symbols
    acsf = Acsf(geo, basis, shell_dict, g1_params=rcut,
                g2_params=torch.tensor([0.5, 1.0]), element_resolve=True)
    acsf()

    # get reference
    acsf_d = ACSF(species=species, rcut=rcut, g2_params=[[0.5, 1.0]])
    acsf_d_g1 = torch.from_numpy(acsf_d.create(ch4))

    # switch last dimension due to the orders of atom specie difference
    assert torch.max(abs(acsf_d_g1[:, [1, 3]] - acsf.g[:, 2:])) < 1E-6


def test_batch_g2(device):
    """Test G2 values in batch geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms([ch4, h2o])
    basis = Basis(geo.atomic_numbers, shell_dict)
    species = [chemical_symbols[ii] for ii in geo.unique_atomic_numbers()]
    acsf = Acsf(geo, basis, shell_dict, g1_params=rcut,
                g2_params=torch.tensor([0.5, 1.0]),
                element_resolve=True, atom_like=False)
    acsf()

    # get reference
    acsf_d = ACSF(species=species, rcut=rcut, g2_params=[[0.5, 1.0]])
    acsf_d_g1 = pack([torch.from_numpy(ii) for ii in acsf_d.create([ch4, h2o])])

    # switch last dimension due to the orders of atom specie difference
    assert torch.max(abs(acsf_d_g1[..., :4] - acsf.g[..., [0, 3, 1, 4]])) < 1E-6


def test_single_g3(device):
    """Test G3 values in single geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms(ch4)
    basis = Basis(geo.atomic_numbers, shell_dict)
    species = geo.chemical_symbols
    acsf = Acsf(geo, basis, shell_dict, g1_params=rcut,
                g3_params=torch.tensor([1.0]),
                element_resolve=True)
    acsf()

    # get reference
    acsf_d = ACSF(species=species, rcut=rcut, g3_params=[1.0])
    acsf_d = torch.from_numpy(acsf_d.create(ch4))

    # switch last dimension due to the orders of atom specie difference
    assert torch.max(abs(acsf_d[:, [1, 3]] - acsf.g[:, 2:])) < 1E-6


def test_batch_g3(device):
    """Test G3 values in batch geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms([ch4, h2o])
    basis = Basis(geo.atomic_numbers, shell_dict)
    species = [chemical_symbols[ii] for ii in geo.unique_atomic_numbers()]
    acsf = Acsf(geo, basis, shell_dict, g1_params=rcut,
                g3_params=torch.tensor([1.0]),
                element_resolve=True, atom_like=False)
    g = acsf()

    # get reference
    acsf_d = ACSF(species=species, rcut=rcut, g3_params=[ 1.0])
    acsf_d = pack([torch.from_numpy(ii) for ii in acsf_d.create([ch4, h2o])])

    # switch last dimension due to the orders of atom specie difference
    assert torch.max(abs(
        acsf_d[..., :4] - g[..., [0, 3, 1, 4]])) < 1E-6, 'tolerance check'


def test_single_g4(device):
    """Test G4 values in single geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms(ch4)
    basis = Basis(geo.atomic_numbers, shell_dict)
    species = geo.chemical_symbols
    acsf = Acsf(geo, basis, shell_dict, g1_params=rcut, g4_params=torch.tensor(
        [[0.02, 1.0, -1.0]]), element_resolve=True, atom_like=False)
    g = acsf()

    acsf_t = AcsfTest(geo, basis, shell_dict, g1_params=rcut)
    acsf_t(g4_params=torch.tensor([0.02, 1.0, -1.0]))

    acsf_d = ACSF(species=species, rcut=rcut, g4_params=[[0.02, 1.0, -1.0]])
    acsf_d_g4 = torch.from_numpy(acsf_d.create(ch4))

    assert torch.max(abs(acsf_d_g4 - g)) < 1E-6, 'tolerance check'


def test_cho_g4(device):
    """Test G4 values in single geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms(cho)
    basis = Basis(geo.atomic_numbers, shell_dict)
    species = geo.chemical_symbols
    acsf = Acsf(geo, basis, shell_dict, g1_params=rcut, g4_params=torch.tensor(
        [[0.02, 1.0, -1.0]]), element_resolve=True)
    g = acsf()

    acsf_t = AcsfTest(geo, basis, shell_dict, g1_params=rcut,
                      element_resolve=True)
    acsf_t(g4_params=torch.tensor([0.02, 1.0, -1.0]))

    acsf_d = ACSF(species=species, rcut=rcut, g4_params=[[0.02, 1.0, -1.0]])
    acsf_d_g4 = torch.from_numpy(acsf_d.create(cho))

    assert torch.max(abs(
        acsf_d_g4[:, 2:].sum(-1) - g[:, 2:].sum(-1))) < 1E-6, 'tolerance check'


def test_batch_g4(device):
    """Test G4 values in batch geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms([ch4, h2o, cho])
    basis = Basis(geo.atomic_numbers, shell_dict)
    acsf = Acsf(geo, basis, shell_dict, g1_params=rcut, g4_params=torch.tensor(
        [[0.02, 1.0, -1.0]]), element_resolve=True, atom_like=False)
    g = acsf()

    acsf_d = ACSF(species=geo.unique_atomic_numbers().numpy(), rcut=rcut,
                  g4_params=[[0.02, 1.0, -1.0]])
    acsf_d_g4 = pack([torch.from_numpy(acsf_d.create(ch4)),
                      torch.from_numpy(acsf_d.create(h2o)),
                      torch.from_numpy(acsf_d.create(cho))])

    assert torch.max(abs(acsf_d_g4[..., 2:].sum(-1) -
                         g[..., 2:].sum(-1))) < 1E-6, 'tolerance check'


def test_single_g5(device):
    """Test G5 values in single geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms(ch4)
    basis = Basis(geo.atomic_numbers, shell_dict)
    species = geo.chemical_symbols
    acsf = Acsf(geo, basis, shell_dict, g1_params=rcut,
                g5_params=torch.tensor([[0.02, 1.0, -1.0]]),
                element_resolve=True)
    acsf()

    acsf2 = Acsf(geo, basis, shell_dict, g1_params=rcut,
                 g5_params=torch.tensor([[0.02, 1.0, -1.0]]),
                  element_resolve=True)
    acsf2()

    # get reference from Dscribe
    acsf_d = ACSF(species=species, rcut=rcut, g5_params=[[0.02, 1.0, -1.0]])
    acsf_d_g5 = torch.from_numpy(acsf_d.create(ch4))

    assert torch.max(abs(acsf_d_g5[..., 2:].sum(-1) - acsf.g[..., 2:].sum(-1))) < 1E-6
    assert torch.max(abs(acsf_d_g5[..., 2:].sum(-1) - acsf2.g[..., 2:].sum(-1))) < 1E-6


def test_batch_g5(device):
    """Test G5 values in batch geometry."""


def test_batch(device):
    """Test G4 values in batch geometry."""
    rcut = 6.0
    g2_params=torch.tensor([0.5, 1.0])
    g4_params=torch.tensor(
        [[0.02, 1.0, -1.0]])
    geo = Geometry.from_ase_atoms([ch4, h2o])
    basis = Basis(geo.atomic_numbers, shell_dict)
    acsf = Acsf(geo, basis, shell_dict, g1_params=rcut, g2_params=g2_params,
                g4_params=g4_params, element_resolve=True)
    acsf()

    acsf_d = ACSF(species=geo.unique_atomic_numbers().numpy(), rcut=rcut,
                  g2_params=[[0.5, 1.0]], g4_params=[[0.02, 1.0, -1.0]])
    acsf_d_g4 = pack([torch.from_numpy(acsf_d.create(ch4)),
                      torch.from_numpy(acsf_d.create(h2o))])

# test_batch(torch.device('cpu'))
