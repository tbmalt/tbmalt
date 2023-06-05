import pytest
from tbmalt.data.units import energy_units, length_units, dipole_units


def test_energy_units():

    energy_values = {
        'rydberg': 2.0, 'ry': 2.0,
        'electronvolt': 27.2113845, 'ev': 27.2113845,
        'kcal/mol': 627.089624894492,
        'kelvin': 315774.6477075367, 'k': 315774.6477075367,
        'cm^-1': 219474.63137054,
        'joule': 4.3597441775e-18, 'j': 4.3597441775e-18,
        'hartree': 1.0, 'ha': 1.0,
        'au': 1.0}

    for key, value in energy_values.items():
        check = value == pytest.approx(1/energy_units[key])
        assert check, f"Energy conversion Ha->{key} failed"


def test_length_units():

    length_values = {
        'angstrom': 0.529177249, 'aa': 0.529177249, 'a': 0.529177249,
        'meter': 1.0e-10 * 0.529177249, 'm': 1.0e-10 * 0.529177249,
        'picometer': 1.0e+2 * 0.529177249, 'pm': 1.0e+2 * 0.529177249,
        'bohr': 1.0,
        'au': 1.0
    }

    for key, value in length_values.items():
        check = value == pytest.approx(1/length_units[key])
        assert check, f"Length conversion A->{key} failed"


def test_dipole_units():
    dipole_values = {
        'cm': 8.47835368557766e-30,
        'coulombmeter': 8.47835368557766e-30,
        'debye': 2.541746674715738, 'd': 2.541746674715738,
        'ebohr': 1.0, 'eb': 1.0,
        'au': 1.0
    }

    for key, value in dipole_values.items():
        check = value == pytest.approx(1/dipole_units[key])
        assert check, f"Dipole conversion A->{key} failed"

