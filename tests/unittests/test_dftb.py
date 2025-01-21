
import pytest

import torch
from ase.build import molecule

from tbmalt import Geometry, OrbitalInfo
from tbmalt.physics.dftb import Dftb1, Dftb2
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed
from tbmalt.common.maths.interpolation import CubicSpline
from tbmalt.common.batch import pack


torch.set_default_dtype(torch.float64)


# Todo:
#   - Gradiant tests should be added once backpropagatable feeds have been
#     implemented.
#   - add more tests for DFTB2 calculations, right now only atomic charges


@pytest.fixture(scope="session")
def shell_resolved_feeds(device, skf_file):
    species = [1, 6, 8, 79]
    h_feed = SkFeed.from_database(skf_file, species, 'hamiltonian', device=device)
    s_feed = SkFeed.from_database(skf_file, species, 'overlap', device=device)
    o_feed = SkfOccupationFeed.from_database(skf_file, species, device=device)

    return h_feed, s_feed, o_feed


@pytest.fixture(scope="session")
def shell_resolved_feeds_scc(device, skf_file):
    species = [1, 6, 8]
    h_feed = SkFeed.from_database(skf_file, species, 'hamiltonian', device=device)
    s_feed = SkFeed.from_database(skf_file, species, 'overlap', device=device)
    o_feed = SkfOccupationFeed.from_database(skf_file, species, device=device)
    u_feed = HubbardFeed.from_database(skf_file, species, device=device)

    return h_feed, s_feed, o_feed, u_feed

@pytest.fixture(scope="session")
def shell_resolved_feeds_scc_spline(device, skf_file):
    species = [1, 6, 8]
    h_feed = SkFeed.from_database(skf_file, species, 'hamiltonian',
                                  interpolation=CubicSpline, device=device)
    s_feed = SkFeed.from_database(skf_file, species, 'overlap',
                                  interpolation=CubicSpline, device=device)
    o_feed = SkfOccupationFeed.from_database(skf_file, species, device=device)
    u_feed = HubbardFeed.from_database(skf_file, species, device=device)

    return h_feed, s_feed, o_feed, u_feed


def H2(device):

    geometry = Geometry(
        torch.tensor([1, 1], device=device),
        torch.tensor([
            [+0.000000000000000E+00, +0.000000000000000E+00, +6.965208740483853E-01],
            [+0.000000000000000E+00, +0.000000000000000E+00, -6.965208740483853E-01]],
            device=device))

    orbs = OrbitalInfo(
        torch.tensor([1, 1], device=device),
        {1: [0]})

    results = {
        'q_final': torch.tensor([
            +1.000000000000000E+00, +1.000000000000000E+00],
            device=device),

        'q_delta_atomic': torch.tensor([
            +2.220446049250310E-16, +0.000000000000000E+00],
            device=device),

        'n_electrons': torch.tensor(+2.000000000000000E+00, device=device),

        'occupancy': torch.tensor([
            +2.000000000000000E+00, +3.275096686248530E-34],
            device=device),

        'eig_values': torch.tensor([
            -3.405911944959140E-01, +2.311892808528270E-01],
            device=device),

        'band_energy': torch.tensor(-6.811823890000001E-01, device=device),

        'band_free_energy': torch.tensor(-6.811823890000001E-01, device=device),

        'fermi_energy': torch.tensor(-5.470095680000000E-02, device=device)
    }

    kwargs = {'filling_scheme': 'fermi', 'filling_temp': 0.0036749324}

    return geometry, orbs, results, kwargs


def H2_scc(device, **kwargs):

    shell_resolved = kwargs.get('shell_resolved', False)

    geometry = Geometry.from_ase_atoms(molecule('H2'), device=device)

    orbs = OrbitalInfo(geometry.atomic_numbers, {1: [0]},
                        shell_resolved=shell_resolved)

    results = {
        'q_final_atomic': torch.tensor([
            +1.000000000000000E+00, +1.000000000000000E+00],
            device=device),
        'band_energy': torch.tensor(-0.6811823890, device=device),
        'core_band_energy': torch.tensor(-0.6811823890, device=device),
        'scc_energy': torch.tensor(0.0000000000, device=device),
    }

    kwargs = {'filling_scheme': 'fermi', 'filling_temp': 0.0036749324}

    return geometry, orbs, results, kwargs

def H2O(device):

    geometry = Geometry(
        torch.tensor([8, 1, 1], device=device),
        torch.tensor([
            [+0.000000000000000E+00, +0.000000000000000E+00, +2.253725008498996E-01],
            [+0.000000000000000E+00, +1.442312573796989E+00, -9.014881136736096E-01],
            [+0.000000000000000E+00, -1.442312573796989E+00, -9.014881136736096E-01]],
            device=device))

    orbs = OrbitalInfo(
        torch.tensor([8, 1, 1], device=device),
        {8: [0, 1], 1: [0]})

    results = {
        'q_final': torch.tensor([
            +1.766409105202590E+00, +1.278899290680150E+00,
            +1.709887433005520E+00, +2.000000000000000E+00,
            +6.224020855558700E-01, +6.224020855558690E-01],
            device=device),

        'q_delta_atomic': torch.tensor([
            +7.551958288882630E-01, -3.775979144441300E-01,
            -3.775979144441310E-01],
            device=device),

        'n_electrons': torch.tensor(+8.000000000000000E+00, device=device),

        'occupancy': torch.tensor([
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +1.349060842183770E-41, +0.000000000000000E+00],
            device=device),

        'eig_values': torch.tensor([
            -9.132557483583750E-01, -4.624227416350380E-01,
            -3.820768853933290E-01, -3.321317735294000E-01,
            +3.646334053157620E-01, +5.759427653670470E-01],
            device=device),

        'band_energy': torch.tensor(-4.179774297800000E+00, device=device),

        'band_free_energy': torch.tensor(-4.179774297800000E+00, device=device),

        'fermi_energy': torch.tensor(+1.625081590000000E-02, device=device)
    }

    kwargs = {'filling_scheme': 'fermi', 'filling_temp': 0.0036749324}

    return geometry, orbs, results, kwargs


def H2O_scc(device, **kwargs):

    shell_resolved = kwargs.get('shell_resolved', False)

    geometry = Geometry.from_ase_atoms(molecule('H2O'), device=device)

    orbs = OrbitalInfo(geometry.atomic_numbers, {1: [0], 8: [0, 1]},
                        shell_resolved=shell_resolved)

    results = {
        'q_final_atomic': torch.tensor([
            6.58558984371061, 0.70720507814469, 0.70720507814469],
            device=device),
        'band_energy': torch.tensor(-3.6852906614, device=device),
        'core_band_energy': torch.tensor(-4.1744311432, device=device),
        'scc_energy': torch.tensor(0.0182663972, device=device),
    }

    kwargs = {'filling_scheme': 'fermi', 'filling_temp': 0.0036749324}

    return geometry, orbs, results, kwargs


def CH4(device):

    geometry = Geometry(
        torch.tensor([6, 1, 1, 1, 1], device=device),
        torch.tensor([
            [+0.000000000000000E+00, +0.000000000000000E+00, +0.000000000000000E+00],
            [+1.188860634482795E+00, +1.188860634482795E+00, +1.188860634482795E+00],
            [-1.188860634482795E+00, -1.188860634482795E+00, +1.188860634482795E+00],
            [+1.188860634482795E+00, -1.188860634482795E+00, -1.188860634482795E+00],
            [-1.188860634482795E+00, +1.188860634482795E+00, -1.188860634482795E+00]],
            device=device))

    orbs = OrbitalInfo(
        torch.tensor([6, 1, 1, 1, 1], device=device),
        {1: [0], 6: [0, 1]})

    results = {
        'q_final': torch.tensor([
            +1.306283024278570E+00, +1.017593294393230E+00,
            +1.017593294393230E+00, +1.017593294393230E+00,
            +9.102342731354360E-01, +9.102342731354360E-01,
            +9.102342731354360E-01, +9.102342731354360E-01],
            device=device),

        'q_delta_atomic': torch.tensor([
            +3.590629074582560E-01, -8.976572686456370E-02,
            -8.976572686456390E-02, -8.976572686456430E-02,
            -8.976572686456430E-02],
            device=device),

        'n_electrons': torch.tensor(+8.000000000000000E+00, device=device),

        'occupancy': torch.tensor([
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +5.668228834912260E-41, +0.000000000000000E+00,
            +0.000000000000000E+00, +0.000000000000000E+00],
            device=device),

        'eig_values': torch.tensor([
            -5.848045161972910E-01, -3.452466354819620E-01,
            -3.452466354819610E-01, -3.452466354819610E-01,
            +3.409680475190840E-01, +3.409680475190840E-01,
            +3.409680475190840E-01, +5.851305739425210E-01],
            device=device),

        'band_energy': torch.tensor(-3.241088845300000E+00, device=device),

        'band_free_energy': torch.tensor(-3.241088845300000E+00, device=device),

        'fermi_energy': torch.tensor(-2.139294000000000E-03, device=device)
    }

    kwargs = {'filling_scheme': 'fermi', 'filling_temp': 0.0036749324}

    return geometry, orbs, results, kwargs


def CH4_scc(device, **kwargs):

    shell_resolved = kwargs.get('shell_resolved', False)

    geometry = Geometry.from_ase_atoms(molecule('CH4'), device=device)

    orbs = OrbitalInfo(geometry.atomic_numbers, {1: [0], 6: [0, 1]},
                        shell_resolved=shell_resolved)

    results = {
        'q_final_atomic': torch.tensor([
            4.30537894059011, 0.92365526485247, 0.92365526485247,
            0.92365526485247, 0.92365526485247],
            device=device),
        'band_energy': torch.tensor(-3.1646777208, device=device),
        'core_band_energy': torch.tensor(-3.2409102228, device=device),
        'scc_energy': torch.tensor(0.0010158495, device=device),
    }

    kwargs = {'filling_scheme': 'fermi', 'filling_temp': 0.0036749324}

    return geometry, orbs, results, kwargs


def CH3O(device):

    geometry = Geometry(
        torch.tensor([6, 8, 1, 1, 1], device=device),
        torch.tensor([
            [+1.299088351396604E+00, +5.598502213763918E-02, -1.549839872273118E-01],
            [-1.270073498567963E+00, -2.374402910129646E-01, +5.745711868274216E-02],
            [-2.063297320629897E+00, +1.398930134277182E+00, -1.798395531550904E-01],
            [+2.121752214634610E+00, +1.842979836799446E+00, +4.270648453369166E-01],
            [+2.307602986159369E+00, -1.678295886072759E+00, +2.230160125421416E-01]],
            device=device))

    orbs = OrbitalInfo(
        torch.tensor([6, 8, 1, 1, 1], device=device),
        {8: [0, 1], 1: [0], 6: [0, 1]})

    results = {
        'q_final': torch.tensor([
            +1.279314077045530E+00, +9.772535254864750E-01,
            +1.144928112565950E+00, +7.763406006069220E-01,
            +1.740425393443830E+00, +1.614269837117830E+00,
            +1.788718143680810E+00, +1.291136554899500E+00,
            +6.098113394312760E-01, +8.846626792636350E-01,
            +8.931397364582450E-01],
            device=device),

        'q_delta_atomic': torch.tensor([
            +1.778363157048840E-01, +4.345499291419620E-01,
            -3.901886605687240E-01, -1.153373207363650E-01,
            -1.068602635417550E-01],
            device=device),

        'n_electrons': torch.tensor(+1.300000000000000E+01, device=device),

        'occupancy': torch.tensor([
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +1.000000000000000E+00, +5.657397638017430E-61,
            +0.000000000000000E+00, +0.000000000000000E+00,
            +0.000000000000000E+00],
            device=device),

        'eig_values': torch.tensor([
            -9.343300049471550E-01, -5.673872360060790E-01,
            -4.452360020405930E-01, -3.983502655691470E-01,
            -3.683978271535110E-01, -3.125648536497800E-01,
            -1.503005758641450E-01, +3.620506858789190E-01,
            +4.193044337046300E-01, +4.627017592047210E-01,
            +5.310120066272410E-01],
            device=device),

        'band_energy': torch.tensor(-6.202832954600000E+00, device=device),

        'band_free_energy': torch.tensor(-6.207927492700000E+00, device=device),

        'fermi_energy': torch.tensor(-1.503005759000000E-01, device=device)
    }

    kwargs = {'filling_scheme': 'fermi', 'filling_temp': 0.0036749324}

    return geometry, orbs, results, kwargs


def CH3O_scc(device, **kwargs):

    shell_resolved = kwargs.get('shell_resolved', False)

    geometry = Geometry.from_ase_atoms(molecule('CH3O'), device=device)

    orbs = OrbitalInfo(geometry.atomic_numbers, {1: [0], 6: [0, 1], 8:[0, 1]},
                        shell_resolved=shell_resolved)

    results = {
        'q_final_atomic': torch.tensor([
            3.91394532144686, 6.31172685711984, 0.90943832879409,
            0.93244474631961, 0.93244474631961], device=device),
        'band_energy': torch.tensor(-5.7884294633, device=device),
        'core_band_energy': torch.tensor(-6.1486854611, device=device),
        'scc_energy': torch.tensor(0.0142612529, device=device),
    }

    kwargs = {'filling_scheme': 'fermi', 'filling_temp': 0.0036749324}

    return geometry, orbs, results, kwargs


def C_wire_scc(device, **kwargs):
    """This test mainly tests polynomial to zero part."""
    shell_resolved = kwargs.get('shell_resolved', False)
    geometry = Geometry(
        torch.tensor([6, 6, 6, 6, 6, 6, 6, 6], device=device),
        torch.tensor([
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.8], [0.0, 0.0, 1.6], [0.0, 0.0, 2.4],
            [0.0, 0.0, 3.2], [0.0, 0.0, 4.0], [0.0, 0.0, 4.8], [0.0, 0.0, 5.6]],
            device=device), units='a')

    orbs = OrbitalInfo(geometry.atomic_numbers, {6: [0, 1]},
                        shell_resolved=shell_resolved)

    results = {
        'q_final_atomic': torch.tensor([
            4.44894528173204, 3.88209725653184, 3.75703400352129, 3.91192345821486,
            3.91192345821486, 3.75703400352129, 3.88209725653184, 4.44894528173204],
            device=device),
        'band_energy': torch.tensor(-18.0811956490, device=device),
        'core_band_energy': torch.tensor(-16.8435335797, device=device),
        'scc_energy': torch.tensor(0.0395935160, device=device),
    }

    kwargs = {'filling_scheme': 'fermi', 'filling_temp': 0.0036749324}

    return geometry, orbs, results, kwargs


def Au13(device):

    geometry = Geometry(
        torch.tensor([79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79], device=device),
        torch.tensor([
            [-9.515615391831613E-01, -9.515615391831613E-01, +0.000000000000000E+00],
            [-9.515615391831613E-01, +0.000000000000000E+00, -9.515615391831613E-01],
            [-9.515615391831613E-01, +0.000000000000000E+00, +9.515615391831613E-01],
            [-9.515615391831613E-01, +9.515615391831613E-01, +0.000000000000000E+00],
            [+0.000000000000000E+00, -9.515615391831613E-01, -9.515615391831613E-01],
            [+0.000000000000000E+00, -9.515615391831613E-01, +9.515615391831613E-01],
            [+9.515615391831613E-01, -9.515615391831613E-01, +0.000000000000000E+00],
            [+0.000000000000000E+00, +9.515615391831613E-01, -9.515615391831613E-01],
            [+9.515615391831613E-01, +0.000000000000000E+00, -9.515615391831613E-01],
            [+0.000000000000000E+00, +0.000000000000000E+00, +0.000000000000000E+00],
            [+0.000000000000000E+00, +9.515615391831613E-01, +9.515615391831613E-01],
            [+9.515615391831613E-01, +0.000000000000000E+00, +9.515615391831613E-01],
            [+9.515615391831613E-01, +9.515615391831613E-01, +0.000000000000000E+00]],
            device=device))

    orbs = OrbitalInfo(
        torch.tensor([79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79], device=device),
        {79: [0, 1, 2]})

    results = {
        'q_final': torch.tensor([
            +7.345507670309031E-01, +1.550313945519920E+00,
            +1.252167670890160E+00, +1.550313945519780E+00,
            +1.517350067220400E+00, +1.207385028438620E+00,
            +1.118114725208160E+00, +1.207385028438620E+00,
            +1.303028354814860E+00, +7.345507670309350E-01,
            +1.252167670890160E+00, +1.550313945519950E+00,
            +1.550313945519750E+00, +1.207385028438620E+00,
            +1.207385028438620E+00, +1.256799947413180E+00,
            +1.517350067220460E+00, +1.164343132609850E+00,
            +7.345507670309350E-01, +1.252167670890160E+00,
            +1.550313945519950E+00, +1.550313945519750E+00,
            +1.207385028438620E+00, +1.207385028438620E+00,
            +1.256799947413170E+00, +1.517350067220460E+00,
            +1.164343132609850E+00, +7.345507670309110E-01,
            +1.550313945519910E+00, +1.252167670890170E+00,
            +1.550313945519770E+00, +1.517350067220390E+00,
            +1.207385028438620E+00, +1.118114725208160E+00,
            +1.207385028438620E+00, +1.303028354814860E+00,
            +7.345507670309940E-01, +1.550313945519840E+00,
            +1.550313945519890E+00, +1.252167670890160E+00,
            +1.207385028438620E+00, +1.517350067220610E+00,
            +1.256799947413180E+00, +1.207385028438620E+00,
            +1.164343132609850E+00, +7.345507670309970E-01,
            +1.550313945519830E+00, +1.550313945519890E+00,
            +1.252167670890160E+00, +1.207385028438620E+00,
            +1.517350067220620E+00, +1.256799947413180E+00,
            +1.207385028438620E+00, +1.164343132609850E+00,
            +7.345507670309110E-01, +1.550313945519900E+00,
            +1.252167670890160E+00, +1.550313945519770E+00,
            +1.517350067220390E+00, +1.207385028438620E+00,
            +1.118114725208160E+00, +1.207385028438620E+00,
            +1.303028354814860E+00, +7.345507670309950E-01,
            +1.550313945519830E+00, +1.550313945519890E+00,
            +1.252167670890160E+00, +1.207385028438620E+00,
            +1.517350067220620E+00, +1.256799947413180E+00,
            +1.207385028438610E+00, +1.164343132609850E+00,
            +7.345507670309340E-01, +1.252167670890160E+00,
            +1.550313945519950E+00, +1.550313945519750E+00,
            +1.207385028438620E+00, +1.207385028438620E+00,
            +1.256799947413170E+00, +1.517350067220460E+00,
            +1.164343132609850E+00, +3.354891096850630E-01,
            +6.347473507869500E-01, +6.347473507869520E-01,
            +6.347473507869500E-01, +6.468722592981720E-01,
            +6.468722592981710E-01, +7.661688315403110E-01,
            +6.468722592981710E-01, +7.661688315405329E-01,
            +7.345507670309950E-01, +1.550313945519830E+00,
            +1.550313945519890E+00, +1.252167670890160E+00,
            +1.207385028438620E+00, +1.517350067220610E+00,
            +1.256799947413180E+00, +1.207385028438620E+00,
            +1.164343132609850E+00, +7.345507670309330E-01,
            +1.252167670890160E+00, +1.550313945519950E+00,
            +1.550313945519750E+00, +1.207385028438620E+00,
            +1.207385028438620E+00, +1.256799947413180E+00,
            +1.517350067220460E+00, +1.164343132609850E+00,
            +7.345507670309060E-01, +1.550313945519920E+00,
            +1.252167670890160E+00, +1.550313945519780E+00,
            +1.517350067220400E+00, +1.207385028438620E+00,
            +1.118114725208160E+00, +1.207385028438620E+00,
            +1.303028354814860E+00],
            device=device),

        'q_delta_atomic': torch.tensor([
            +4.406095330814180E-01, +4.406095330815170E-01,
            +4.406095330815250E-01, +4.406095330813940E-01,
            +4.406095330817590E-01, +4.406095330817610E-01,
            +4.406095330813880E-01, +4.406095330817500E-01,
            +4.406095330815100E-01, -5.287314396978730E+00,
            +4.406095330817480E-01, +4.406095330815170E-01,
            +4.406095330814200E-01],
            device=device),

        'n_electrons': torch.tensor(+1.430000000000000E+02, device=device),

        'occupancy': torch.tensor([
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +2.000000000000000E+00,
            +2.000000000000000E+00, +5.000000000025530E-01,
            +4.999999999975680E-01, +4.323230522445350E-171,
            +0.000000000000000E+00, +0.000000000000000E+00,
            +0.000000000000000E+00, +0.000000000000000E+00,
            +0.000000000000000E+00, +0.000000000000000E+00,
            +0.000000000000000E+00, +0.000000000000000E+00,
            +0.000000000000000E+00, +0.000000000000000E+00,
            +0.000000000000000E+00, +0.000000000000000E+00,
            +0.000000000000000E+00, +0.000000000000000E+00,
            +0.000000000000000E+00, +0.000000000000000E+00,
            +0.000000000000000E+00, +0.000000000000000E+00,
            +0.000000000000000E+00, +0.000000000000000E+00,
            +0.000000000000000E+00, +0.000000000000000E+00,
            +0.000000000000000E+00, +0.000000000000000E+00,
            +0.000000000000000E+00, +0.000000000000000E+00,
            +0.000000000000000E+00, +0.000000000000000E+00,
            +0.000000000000000E+00, +0.000000000000000E+00,
            +0.000000000000000E+00, +0.000000000000000E+00,
            +0.000000000000000E+00, +0.000000000000000E+00,
            +0.000000000000000E+00, +0.000000000000000E+00,
            +0.000000000000000E+00, +0.000000000000000E+00,
            +0.000000000000000E+00, +0.000000000000000E+00,
            +0.000000000000000E+00, +0.000000000000000E+00,
            +0.000000000000000E+00],
            device=device),

        'eig_values': torch.tensor([
            -1.908639295117550E+00, -1.908639295117550E+00,
            -1.848095305550100E+00, -1.848095305550100E+00,
            -1.848095305550100E+00, -1.837396386339710E+00,
            -1.744816882274070E+00, -1.744816882274070E+00,
            -1.744816882274070E+00, -1.609493952031010E+00,
            -1.575904035766130E+00, -1.575904035766130E+00,
            -1.575904035766130E+00, -1.385173350994070E+00,
            -1.385173350994070E+00, -1.385173350994070E+00,
            -1.335239946968880E+00, -1.335239946968880E+00,
            -1.335239946968870E+00, -1.228779350928200E+00,
            -1.228779350928200E+00, -1.228779350928200E+00,
            -1.126516722801380E+00, -7.202768085135610E-01,
            -7.202768085135590E-01, -5.882901999637270E-01,
            -5.882901999637240E-01, -5.882901999637220E-01,
            -4.375766354282910E-01, -4.375766354282890E-01,
            -3.092178637105160E-01, -3.013679637745180E-01,
            -3.013679637745170E-01, -3.013679637745150E-01,
            -1.794791510906370E-01, -1.505831660693780E-01,
            -1.505831660693760E-01, -5.494341826301490E-02,
            -5.494341826301350E-02, -5.494341826301290E-02,
            +6.319738908206250E-03, +6.319738908206850E-03,
            +6.319738908209990E-03, +1.607978909736500E-01,
            +1.607978909736530E-01, +1.607978909736540E-01,
            +1.788112121622660E-01, +2.231704820917360E-01,
            +2.231704820917370E-01, +2.231704820917380E-01,
            +2.707335757981880E-01, +2.707335757981900E-01,
            +2.707335757981920E-01, +3.376248189174830E-01,
            +3.376248189174830E-01, +8.253138112949620E-01,
            +8.253138112949630E-01, +8.253138112949640E-01,
            +1.194754796718070E+00, +1.194754796718080E+00,
            +1.194754796718080E+00, +4.283254115682630E+00,
            +4.292530749346850E+00, +4.292530749346860E+00,
            +4.292530749346870E+00, +4.796687882324210E+00,
            +4.796687882324230E+00, +4.796687882324250E+00,
            +5.145881543950700E+00, +5.145881543950710E+00,
            +5.145881543950720E+00, +5.287135749171270E+00,
            +5.287135749171320E+00, +6.727241000613440E+00,
            +6.727241000613470E+00, +6.727241000613490E+00,
            +6.789496573642130E+00, +6.789496573642140E+00,
            +6.789496573642170E+00, +6.875637589040720E+00,
            +7.240103529253490E+00, +7.240103529253510E+00,
            +7.810483048056300E+00, +7.810483048056330E+00,
            +7.810483048056360E+00, +8.225115345032910E+00,
            +8.225115345032950E+00, +8.225115345033100E+00,
            +8.291799423821409E+00, +8.291799423821461E+00,
            +8.291799423821480E+00, +9.104389868819620E+00,
            +9.280922440307251E+00, +9.280922440307281E+00,
            +9.280922440307309E+00, +9.891392082963099E+00,
            +9.891392082963121E+00, +9.891392082963289E+00,
            +1.026488713049180E+01, +1.026488713049180E+01,
            +1.122892396498530E+01, +1.122892396498540E+01,
            +1.122892396498560E+01, +1.448998451508210E+01,
            +1.448998451508210E+01, +1.448998451508210E+01,
            +1.721505773402520E+01, +1.721505773402530E+01,
            +1.739360205229830E+01, +1.739360205229840E+01,
            +1.786789540972920E+01, +1.786789540972920E+01,
            +1.786789540972920E+01, +1.979335863588580E+01,
            +1.979335863588590E+01, +1.979335863588590E+01,
            +2.062032262075740E+01],
            device=device),

        'band_energy': torch.tensor(+3.369073400960000E+01, device=device),

        'band_free_energy': torch.tensor(+3.368246783500000E+01, device=device),

        'fermi_energy': torch.tensor(+5.283098423300000E+00, device=device)
    }

    kwargs = {'filling_scheme': 'fermi', 'filling_temp': 0.0036749324}

    return geometry, orbs, results, kwargs


def merge_systems(device, *systems):
    """Combine multiple test systems into a batch."""

    geometry, orbs, results, kwargs = systems[0](device)

    results = {k: [v] for k, v in results.items()}

    for system in systems[1:]:
        t_geometry, t_orbs, t_results, t_kwargs = system(device)

        assert t_kwargs == kwargs, 'Test systems with different settings ' \
                                   'cannot be used together'

        geometry += t_geometry
        orbs += t_orbs

        for k, v in t_results.items():
            results[k].append(t_results[k])

    results = {k: pack(v) for k, v in results.items()}

    return geometry, orbs, results, kwargs


def merge_systems_shell_resolved(device, *systems):
    """Combine multiple test systems into a batch."""

    geometry, orbs, results, kwargs = systems[0](device, shell_resolved=True)

    results = {k: [v] for k, v in results.items()}

    for system in systems[1:]:
        t_geometry, t_orbs, t_results, t_kwargs = system(device,
                                                          shell_resolved=True)

        assert t_kwargs == kwargs, 'Test systems with different settings ' \
                                   'cannot be used together'

        geometry += t_geometry
        orbs += t_orbs

        for k, v in t_results.items():
            results[k].append(t_results[k])

    results = {k: pack(v) for k, v in results.items()}

    return geometry, orbs, results, kwargs


def test_dftb1_general(device, shell_resolved_feeds):
    h_feed, s_feed, o_feed = shell_resolved_feeds
    geometry, orbs, results, kwargs = CH3O(device)

    # Instantiate a blank calculator
    calculator = Dftb1(h_feed, s_feed, o_feed, **kwargs)

    # SECTION 1: Attributes

    # Identify all calculator attributes before initialisation
    pre_init_attr = [k for k in calculator.__dict__.keys()]

    # Those of which that are not yet initialised (excluding the geometry and
    # orbs attributes)
    pre_init_null_attr = [
        k for k, v in calculator.__dict__.items()
        if v is None and k not in ['_geometry', '_orbs']]

    # Run the calculator
    _ = calculator(geometry, orbs)

    # Check 1.1
    # See if any new attributes have been created that were not there before.
    # This prevents new attributes being arbitrarily defined outside of the
    # __init__ function.
    post_init_attr = [k for k in calculator.__dict__.keys()]

    symdiff = set(pre_init_attr) ^ set(post_init_attr)
    check_1_1 = len(symdiff) == 0
    assert check_1_1, 'new/existing attributes should not be defined/destroyed ' \
                      f'outside of __init__, offending attributes are {symdiff}'

    # Check 1.2
    # Ensure that system specific attributes are set back to None upon reset
    calculator.reset()
    failed_reset_attrs = [
        k for k, v in calculator.__dict__.items()
        if k in pre_init_null_attr and v is not None]

    check_1_2 = len(failed_reset_attrs) == 0

    assert check_1_2, f'System specific attribute reset failed: {failed_reset_attrs}'

    # Check 1.3
    # Check that cached attributes are operating as intended and are actively
    # caching
    cached_attrs = ['overlap', 'hamiltonian']
    for c_attr in cached_attrs:
        old_value = calculator.__getattribute__(c_attr)

        # Ensure that the attribute `c_attr` is cached in an underlying
        # attribute named `_c_attr'.
        calculator.__setattr__(f'_{c_attr}', 10.0)
        check_1_3_a = calculator.__getattribute__(c_attr) == 10.0

        # Check that the setter has also been defined correctly.
        calculator.__setattr__(c_attr, 20.0)
        check_1_3_b = calculator.__getattribute__(c_attr) == 20.0

        check_1_3 = check_1_3_a and check_1_3_b

        assert check_1_3, f'Attribute {c_attr} is not properly cached.'

        calculator.__setattr__(c_attr, old_value)


def dftb1_helper(calculator, geometry, orbs, results):

    # Trigger the calculation
    _ = calculator(geometry, orbs)

    # Ensure that the `hamiltonian` and `overlap` properties return the correct
    # matrices. We do not need to actually check if the matrices are themselves
    # correct as this is something that is something that is done by the unit
    # tests for those feeds. Furthermore, any errors in said matrix will cause
    # many of the computed properties to be incorrect.

    # TODO: check hamiltonian matrix
    # TODO: check overlap matrix

    def check_allclose(i):
        predicted = getattr(calculator, i)
        is_close = torch.allclose(predicted, results[i])
        assert is_close, f'Attribute {i} is in error for system {geometry}'
        if isinstance(predicted, torch.Tensor):
            device_check = predicted.device == calculator.device
            assert device_check, f'Attribute {i} was returned on the wrong device'


    check_allclose('q_final')
    check_allclose('q_delta_atomic')
    # `q_zero_atomic` & `q_final_atomic` are used to construct `q_delta_atomic`
    # and thus do not separate tests of their own.

    check_allclose('fermi_energy')
    check_allclose('n_electrons')
    check_allclose('occupancy')

    check_allclose('band_energy')
    check_allclose('band_free_energy')

    check_allclose('eig_values')

    # Rho is not tested here as it is used to construct other properties and so
    # any errors should be caught elsewhere.


def dftb2_helper(calculator, geometry, orbs, results):

    # Trigger the calculation
    _ = calculator(geometry, orbs)

    # Ensure that the `hamiltonian` and `overlap` properties return the correct
    # matrices. We do not need to actually check if the matrices are themselves
    # correct as this is something that is something that is done by the unit
    # tests for those feeds. Furthermore, any errors in said matrix will cause
    # many of the computed properties to be incorrect.

    # TODO: check hamiltonian matrix
    # TODO: check overlap matrix

    def check_allclose(i):
        predicted = getattr(calculator, i)
        is_close = torch.allclose(predicted, results[i])
        assert is_close, f'Attribute {i} is in error for system {geometry}'
        if isinstance(predicted, torch.Tensor):
            device_check = predicted.device == calculator.device
            assert device_check, f'Attribute {i} was returned on the wrong device'

    check_allclose('q_final_atomic')
    check_allclose('band_energy')
    check_allclose('core_band_energy')
    check_allclose('scc_energy')


def test_dftb1_single(device, shell_resolved_feeds):
    h_feed, s_feed, o_feed = shell_resolved_feeds

    systems = [H2, H2O, CH4, CH3O, Au13]

    for system in systems:
        geometry, orbs, results, kwargs = system(device)

        calculator = Dftb1(h_feed, s_feed, o_feed, **kwargs)

        dftb1_helper(calculator, geometry, orbs, results)


def test_dftb1_batch(device, shell_resolved_feeds):
    h_feed, s_feed, o_feed = shell_resolved_feeds

    batches = [[H2], [H2, H2O], [H2, Au13], [Au13, H2],
               [H2, H2O, CH4, CH3O, Au13], [Au13, CH3O, CH4, H2, H2O]]

    for batch in batches:
        geometry, orbs, results, kwargs = merge_systems(device, *batch)

        calculator = Dftb1(h_feed, s_feed, o_feed, **kwargs)
        assert calculator.device == device, 'Calculator is on the wrong device'

        dftb1_helper(calculator, geometry, orbs, results)


def test_dftb2_single(device, shell_resolved_feeds_scc):
    h_feed, s_feed, o_feed, u_feed = shell_resolved_feeds_scc

    systems = [H2_scc, H2O_scc, CH4_scc, CH3O_scc, C_wire_scc]

    for system in systems:
        geometry, orbs, results, kwargs = system(device)

        calculator = Dftb2(h_feed, s_feed, o_feed, u_feed, **kwargs)

        dftb2_helper(calculator, geometry, orbs, results)


def test_dftb2_batch(device, shell_resolved_feeds_scc):
    h_feed, s_feed, o_feed, u_feed = shell_resolved_feeds_scc

    batches = [[H2_scc], [H2_scc, CH3O_scc, C_wire_scc],
               [H2_scc, H2O_scc, CH4_scc, CH3O_scc, C_wire_scc]]

    for batch in batches:
        geometry, orbs, results, kwargs = merge_systems(device, *batch)

        calculator = Dftb2(h_feed, s_feed, o_feed, u_feed, **kwargs)
        assert calculator.device == device, 'Calculator is on the wrong device'

        dftb2_helper(calculator, geometry, orbs, results)


def test_dftb2_single_shell_resolved(device, shell_resolved_feeds_scc):
    h_feed, s_feed, o_feed, u_feed = shell_resolved_feeds_scc

    systems = [H2_scc, H2O_scc, CH4_scc, CH3O_scc, C_wire_scc]

    for system in systems:
        geometry, orbs, results, kwargs = system(device, shell_resolved=True)

        calculator = Dftb2(h_feed, s_feed, o_feed, u_feed, **kwargs)

        dftb2_helper(calculator, geometry, orbs, results)


def test_dftb2_batch_shell_resolved(device, shell_resolved_feeds_scc):
    h_feed, s_feed, o_feed, u_feed = shell_resolved_feeds_scc

    batches = [[H2_scc], [H2_scc, CH3O_scc, C_wire_scc],
               [H2_scc, H2O_scc, CH4_scc, CH3O_scc, C_wire_scc]]

    for batch in batches:
        geometry, orbs, results, kwargs = merge_systems_shell_resolved(device,
                                                                        *batch)

        calculator = Dftb2(h_feed, s_feed, o_feed, u_feed, **kwargs)
        assert calculator.device == device, 'Calculator is on the wrong device'

        dftb2_helper(calculator, geometry, orbs, results)


def test_dftb2_batch_spl(device, shell_resolved_feeds_scc_spline):
    h_feed, s_feed, o_feed, u_feed = shell_resolved_feeds_scc_spline

    batches = [[H2_scc], [H2_scc, CH3O_scc, C_wire_scc],
               [H2_scc, H2O_scc, CH4_scc, CH3O_scc, C_wire_scc]]

    for batch in batches:
        geometry, orbs, results, kwargs = merge_systems(device, *batch)

        calculator = Dftb2(h_feed, s_feed, o_feed, u_feed, **kwargs)
        assert calculator.device == device, 'Calculator is on the wrong device'

        dftb2_helper(calculator, geometry, orbs, results)
