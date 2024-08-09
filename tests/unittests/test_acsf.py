#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acsf module unit-tests."""
import torch
import pytest
from importlib.metadata import version
from packaging.version import parse as parse_version
from dscribe.descriptors import ACSF
from ase.build import molecule
from tbmalt.ml.acsf import Acsf
from tbmalt import Geometry
from tbmalt.data.elements import chemical_symbols
from tbmalt.common.batch import pack

dscribe_version = parse_version(version('dscribe'))

# TODO: resolve describe version update issue
if dscribe_version <= parse_version("1.2.2"):
    pytestmark = pytest.mark.skip(
        "Skipping tests: Deprecated dscribe package detected")


# Set some global parameters which only used here
torch.set_default_dtype(torch.float64)
ch4 = molecule('CH4')
nh3 = molecule('NH3')
h2o = molecule('H2O')
h2o2 = molecule('H2O2')
h2 = molecule('H2')
cho = molecule('CH3CHO')
text = 'tolerance check'
textd = 'Device persistence check'


def test_single_g1(device):
    """Test G1 values in single geometry."""
    rcut = 6.0

    # 1. Molecule test, test for element resolved
    geo = Geometry.from_ase_atoms(ch4, device=device)
    species = geo.chemical_symbols
    acsf = Acsf(geo, g1_params=rcut, element_resolve=True)
    acsf()

    # Get reference
    acsf_t = ACSF(rcut, species=species)
    acsf_t_g = torch.tensor(acsf_t.create(ch4), device=device)

    assert torch.max(abs(acsf_t_g - acsf.g)) < 1E-6, text
    assert acsf.g.device == device, textd

    acsf_sum = Acsf(geo, g1_params=rcut, element_resolve=False)
    acsf_sum()
    assert torch.max(abs(acsf_t_g.sum(-1) - acsf_sum.g)) < 1E-6, text
    assert acsf_sum.g.device == device, textd

    # 2. Periodicity system test
    ch4.cell = [1, 3, 3]
    geo = Geometry.from_ase_atoms(ch4, device=device)
    species = geo.chemical_symbols
    acsfp = Acsf(geo, g1_params=rcut, element_resolve=True)
    acsfp()

    # Get reference
    acsf_tp = ACSF(rcut, species=species, periodic=True)
    acsf_t_tp = torch.from_numpy(acsf_tp.create(ch4)).to(device)

    assert torch.max(abs(acsf_t_tp - acsfp.g)) < 1E-6, text
    assert acsfp.g.device == device, textd


def test_batch_g1(device):
    """Test G1 values in batch geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms([ch4, h2o], device=device)
    species = [chemical_symbols[ii] for ii in geo.unique_atomic_numbers()]
    acsf = Acsf(geo, g1_params=rcut)
    acsf()

    # get reference
    acsf_d = ACSF(rcut, species=species)
    acsf_d_g1 = [torch.from_numpy(ii).to(device) for ii in
        acsf_d.create([ch4, h2o])]


    assert torch.max(abs(acsf_d_g1[0] -
                         acsf.g[: acsf_d_g1[0].shape[0]])) < 1E-6, text
    assert torch.max(abs(acsf_d_g1[1] -
                         acsf.g[acsf_d_g1[0].shape[0]:])) < 1E-6, text
    assert acsf.g.device == device, textd

    acsf_sum = Acsf(geo, g1_params=rcut, element_resolve=False)
    acsf_sum()
    assert torch.max(abs(torch.cat([
        ii for ii in acsf_d_g1]).sum(-1) - acsf_sum.g)) < 1E-6, text
    assert acsf_sum.g.device == device, textd


def test_single_g2(device):
    """Test G2 values in single geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms(ch4, device=device)
    species = geo.chemical_symbols
    acsf = Acsf(geo, g1_params=rcut,
                g2_params=torch.tensor([0.5, 1.0]), element_resolve=True)
    acsf()

    # get reference
    acsf_d = ACSF(rcut, species=species, g2_params=[[0.5, 1.0]])
    acsf_d_g = torch.from_numpy(acsf_d.create(ch4)).to(device)

    # element resolved True & select G2
    assert torch.max(abs(acsf_d_g[:, [1, 3]] - acsf.g[:, 2:])) < 1E-6, text
    assert acsf.g.device == device, textd

    # element resolved False & G1 + G2
    acsf_sum = Acsf(geo, g1_params=rcut,
            g2_params=torch.tensor([0.5, 1.0]), element_resolve=False)
    acsf_sum()
    assert torch.max(abs(acsf_d_g.sum(-1) - acsf_sum.g.sum(-1))) < 1E-6, text
    assert acsf_sum.g.device == device, textd


def test_batch_g2(device):
    """Test G2 values in batch geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms([ch4, h2o], device=device)
    species = [chemical_symbols[ii] for ii in geo.unique_atomic_numbers()]
    acsf = Acsf(geo, g1_params=rcut,
                g2_params=torch.tensor([0.5, 1.0]),
                element_resolve=True, atom_like=False)
    acsf()

    # get reference
    acsf_d = ACSF(rcut, species=species, g2_params=[[0.5, 1.0]])
    acsf_d_g1 = pack([torch.from_numpy(ii) for ii in
        acsf_d.create([ch4, h2o])]).to(device)

    # switch last dimension due to the orders of atom specie difference
    assert torch.max(abs(acsf_d_g1[..., :4] - acsf.g[..., [0, 3, 1, 4]])) < 1E-6, text
    assert acsf.g.device == device, textd


def test_single_g3(device):
    """Test G3 values in single geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms(ch4, device=device)
    species = geo.chemical_symbols
    acsf = Acsf(geo, g1_params=rcut,
                g3_params=torch.tensor([1.0]),
                element_resolve=True)
    acsf()

    # get reference
    acsf_d = ACSF(rcut, species=species, g3_params=[1.0])
    acsf_d = torch.from_numpy(acsf_d.create(ch4)).to(device)

    # switch last dimension due to the orders of atom specie difference
    assert torch.max(abs(acsf_d[:, [1, 3]] - acsf.g[:, 2:])) < 1E-6, text
    assert acsf.g.device == device, textd


def test_batch_g3(device):
    """Test G3 values in batch geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms([ch4, h2o], device=device)
    species = [chemical_symbols[ii] for ii in geo.unique_atomic_numbers()]
    acsf = Acsf(geo, g1_params=rcut,
                g3_params=torch.tensor([1.0]),
                element_resolve=True, atom_like=False)
    g = acsf()

    # get reference
    acsf_d = ACSF(rcut, species=species, g3_params=[ 1.0])
    acsf_d = pack([torch.from_numpy(ii) for ii in
        acsf_d.create([ch4, h2o])]).to(device)

    # switch last dimension due to the orders of atom specie difference
    assert torch.max(abs(
        acsf_d[..., :4] - g[..., [0, 3, 1, 4]])) < 1E-6, text
    assert g.device == device, textd


@pytest.mark.skip(reason="Element resolved ACSF, as required by TBMaLT, are not implemented in G4 and G5.")
def test_single_g4(device):
    """Test G4 values in single geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms(ch4, device=device)
    species = geo.chemical_symbols
    acsf = Acsf(geo, g1_params=rcut, g4_params=torch.tensor(
        [[0.02, 1.0, -1.0]]), element_resolve=True, atom_like=False)
    g = acsf()

    acsf_d = ACSF(rcut, species=species, g4_params=[[0.02, 1.0, -1.0]])
    acsf_d_g4 = torch.from_numpy(acsf_d.create(ch4)).to(device)

    assert torch.max(abs(acsf_d_g4 - g)) < 1E-6, text
    assert g.device == device, textd


@pytest.mark.skip(reason="Element resolved ACSF, as required by TBMaLT, are not implemented in G4 and G5.")
def test_cho_g4(device):
    """Test G4 values in single geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms(cho, device=device)
    species = geo.chemical_symbols
    acsf = Acsf(geo, g1_params=rcut, g4_params=torch.tensor(
        [[0.02, 1.0, -1.0]]), element_resolve=True)
    g = acsf()

    acsf_d = ACSF(rcut, species=species, g4_params=[[0.02, 1.0, -1.0]])
    acsf_d_g4 = torch.from_numpy(acsf_d.create(cho)).to(device)

    assert torch.max(abs(
        acsf_d_g4[:, 2:].sum(-1) - g[:, 2:].sum(-1))) < 1E-6, text
    assert g.device == device, textd


@pytest.mark.skip(reason="Element resolved ACSF, as required by TBMaLT, are not implemented in G4 and G5.")
def test_batch_g4(device):
    """Test G4 values in batch geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms([ch4, h2o, cho], device=device)
    acsf = Acsf(geo, g1_params=rcut, g4_params=torch.tensor(
        [[0.02, 1.0, -1.0]]), element_resolve=True, atom_like=False)
    g = acsf()
    uniq = geo.unique_atomic_numbers()
    uniq = uniq.numpy() if device == torch.device('cpu') else uniq.cpu().numpy()

    acsf_d = ACSF(rcut, species=uniq, g4_params=[[0.02, 1.0, -1.0]])
    acsf_d_g4 = pack([torch.from_numpy(acsf_d.create(ch4)).to(device),
                      torch.from_numpy(acsf_d.create(h2o)).to(device),
                      torch.from_numpy(acsf_d.create(cho)).to(device)])

    assert torch.max(abs(acsf_d_g4[..., 2:].sum(-1) -
                         g[..., 2:].sum(-1))) < 1E-6, text
    assert g.device == device, textd


@pytest.mark.skip(reason="Element resolved ACSF, as required by TBMaLT, are not implemented in G4 and G5.")
def test_single_g5(device):
    """Test G5 values in single geometry."""
    rcut = 6.0
    geo = Geometry.from_ase_atoms(ch4, device=device)
    species = geo.chemical_symbols
    acsf = Acsf(geo, g1_params=rcut,
                g5_params=torch.tensor([[0.02, 1.0, -1.0]]),
                element_resolve=True)
    acsf()

    # get reference from Dscribe
    acsf_d = ACSF(rcut, species=species, g5_params=[[0.02, 1.0, -1.0]])
    acsf_d_g5 = torch.from_numpy(acsf_d.create(ch4)).to(device)

    assert torch.max(abs(
        acsf_d_g5[..., 2:].sum(-1) - acsf.g[..., 2:].sum(-1))) < 1E-6, text
    assert acsf.g.device == device, textd


@pytest.mark.skip(reason="Element resolved ACSF, as required by TBMaLT, are not implemented in G4 and G5.")
def test_batch_g5(device):
    """Test G5 values in batch geometry."""
    rcut = 6.0

    geo = Geometry.from_ase_atoms([ch4, h2o, cho], device=device)
    acsf = Acsf(geo, g1_params=rcut, g5_params=torch.tensor(
        [[0.02, 1.0, -1.0]]), element_resolve=True, atom_like=False)
    g = acsf()
    uniq = geo.unique_atomic_numbers()
    uniq = uniq.numpy() if device == torch.device('cpu') else uniq.cpu().numpy()

    acsf_d = ACSF(rcut, species=uniq,
                  g5_params=[[0.02, 1.0, -1.0]])
    acsf_d_g5 = pack([torch.from_numpy(acsf_d.create(ch4)).to(device),
                      torch.from_numpy(acsf_d.create(h2o)).to(device),
                      torch.from_numpy(acsf_d.create(cho)).to(device)])

    assert torch.max(abs(acsf_d_g5[..., 2:].sum(-1) -
                         g[..., 2:].sum(-1))) < 1E-6, text
    assert g.device == device, textd

