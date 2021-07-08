import numpy as np
from scipy.special import erfc
import pytest
import torch
from torch.autograd import gradcheck
from tbmalt.physics.electronics import (
    get_smearing_function, fermi_smearing, gaussian_smearing, Schemes,
    fermi_search, smearing_entropy
)
from tbmalt.common.batch import pack
from tbmalt import Basis

torch.set_default_dtype(torch.float64)

# Suppress numpy floating point overflow warnings
np.seterr(over='ignore')


#############
# Test Data #
#############
def H2(device):
    e_vals = torch.tensor([-0.3405911944959140, 0.2311892808528265], device=device)
    kt = torch.tensor(0.0036749324000000, device=device)
    e_fermi = {'fermi': torch.tensor(-0.0547009568215437, device=device),
               'gaussian': torch.tensor(-0.0547009568215437, device=device)}
    entropy = {'fermi': torch.tensor(0.0000000000000000, device=device),
               'gaussian': torch.tensor(0.0000000000000000, device=device)}
    n_elec = 2.0

    return e_vals, kt, e_fermi, entropy, n_elec


def H2O(device):
    e_vals = torch.tensor(
        [-0.9132557483583752, -0.4624227416350384, -0.3820768853933298, -0.3321317735293993, 0.3646334053157624,
         0.5759427653670470], device=device)
    kt = torch.tensor(0.0036749324000000, device=device)
    e_fermi = {'fermi': torch.tensor(0.0162508158931816, device=device),
               'gaussian': torch.tensor(0.0162508158931816, device=device)}
    entropy = {'fermi': torch.tensor(0.0000000000000000, device=device),
               'gaussian': torch.tensor(0.0000000000000000, device=device)}
    n_elec = 8.0

    return e_vals, kt, e_fermi, entropy, n_elec


def CH4(device):
    e_vals = torch.tensor(
        [-0.5848045161972909, -0.3452466354819613, -0.3452466354819613, -0.3452466354819613, 0.3409680475190836,
         0.3409680475190837, 0.3409680475190838, 0.5851305739425205], device=device)
    kt = torch.tensor(0.0036749324000000, device=device)
    e_fermi = {'fermi': torch.tensor(-0.0021392939814388, device=device),
               'gaussian': torch.tensor(-0.0021392939814388, device=device)}
    entropy = {'fermi': torch.tensor(0.0000000000000000, device=device),
               'gaussian': torch.tensor(0.0000000000000000, device=device)}
    n_elec = 8.0

    return e_vals, kt, e_fermi, entropy, n_elec


def H2COH(device):
    e_vals = torch.tensor(
        [-0.9343300049471552, -0.5673872360060783, -0.4452360020405934, -0.3983502655691470, -0.3683978271535110,
         -0.3125648536497793, -0.1503005758641445, 0.3620506858789189, 0.4193044337046300, 0.4627017592047214,
         0.5310120066272412], device=device)
    kt = torch.tensor(0.0036749324000000, device=device)
    e_fermi = {'fermi': torch.tensor(-0.1503005758641444, device=device),
               'gaussian': torch.tensor(-0.1503005758641444, device=device)}
    entropy = {'fermi': torch.tensor(0.0025472690318084, device=device),
               'gaussian': torch.tensor(0.0010366792901611, device=device)}
    n_elec = 13.0

    return e_vals, kt, e_fermi, entropy, n_elec


def Au13(device):
    e_vals = torch.tensor(
        [-0.3527299502166848, -0.3446043323626678, -0.3446043323626677, -0.3446043323626675, -0.3396680431031974,
         -0.3396680431031973, -0.3348665076928592, -0.3236859056397957, -0.3137384732690163, -0.3137384732690163,
         -0.3137384732690162, -0.3130603572759072, -0.3130603572759071, -0.3130603572759070, -0.3103190418651061,
         -0.3062146739029074, -0.3062146739029073, -0.3062146739029073, -0.2987540580770446, -0.2987540580770445,
         -0.2891664255726048, -0.2891664255726047, -0.2891664255726045, -0.2886466040614372, -0.2886466040614371,
         -0.2886466040614370, -0.2772119188018515, -0.2772119188018514, -0.2772119188018511, -0.2717068280935509,
         -0.2717068280935508, -0.2717068280935507, -0.2680562588133010, -0.2680562588133009, -0.2680562588133009,
         -0.2508237307023709, -0.2508237307023708, -0.2418141387259901, -0.2418141387259900, -0.2418141387259897,
         -0.2288866289513421, -0.2288866289513419, -0.2288866289513417, -0.2178897159707643, -0.2178897159707643,
         -0.2135507437165386, -0.2135507437165383, -0.2135507437165381, -0.2061925336110552, -0.2061925336110551,
         -0.2061925336110549, -0.2053032768112894, -0.2053032768112893, -0.2053032768112892, -0.2036628759403381,
         -0.2036628759403379, -0.2013005590857649, -0.2013005590857647, -0.2013005590857646, -0.2009927281347400,
         -0.1947910664667635, -0.1947910664667633, -0.1947910664667631, -0.1827849468699932, -0.1827849468699929,
         -0.1827849468699928, -0.1750536501383871, -0.1750536501383868, -0.1459052677523637, -0.1459052677523636,
         -0.1459052677523634, -0.1408475250502909, -0.1373494312542527, -0.1373494312542526, -0.0821103630861045,
         -0.0631299548293009, -0.0631299548293008, -0.0631299548293007, -0.0461239690150328, -0.0301246652659303,
         -0.0301246652659300, -0.0301246652659290, 0.0103136738367398, 0.0103136738367401, 0.0103136738367404,
         0.0519917830212956, 0.0519917830212958, 0.0896344847999815, 0.0896344847999818, 0.0896344847999822,
         0.0936007927691514, 0.0936007927691517, 0.0936007927691523, 0.1583778013667147, 0.1583778013667151,
         0.1583778013667153, 0.1899425534402224, 0.1899425534402227, 0.1899425534402230, 0.1943635284217193,
         0.2170962394457910, 0.2170962394457913, 0.2170962394457915, 0.2255573749871409, 0.2255573749871411,
         0.2769052785968288, 0.2769052785968295, 0.2769052785968298, 0.2951054123540366, 0.2951054123540370,
         0.2951054123540371, 0.3129578834708936, 0.3129578834708938, 0.3129578834708942, 0.3525297937927357,
         0.3589271597154700, 0.3589271597154700], device=device)
    kt = torch.tensor(0.0036749324000000, device=device)
    e_fermi = {'fermi': torch.tensor(-0.1406917041267987, device=device),
               'gaussian': torch.tensor(-0.1411394541869537, device=device)}
    entropy = {'fermi': torch.tensor(0.0123986527453900, device=device),
               'gaussian': torch.tensor(0.0023244775970756, device=device)}
    n_elec = 143.0

    return e_vals, kt, e_fermi, entropy, n_elec


#########################
# get_smearing_function #
#########################
def test_get_smearing_function():
    """Ensures `get_smearing_function` returns the expected result.

    Tests that:
        1) The correct function is returned.
        2) An error is raised when an unexpected name is encountered.
        3) The `Schemes` type is updated when a new method is added.
    """
    schemes = {'fermi': fermi_smearing,
               'gaussian': gaussian_smearing}

    # Check 1
    for name, expected_function in schemes.items():
        if (func := get_smearing_function(name)) is not expected_function:
            pytest.fail(f'Expected "{expected_function}" to be returned, '
                        f'got "{func}" instead.')

    # Check 2
    with pytest.raises(KeyError):
        get_smearing_function('not_a_real_smearing_function_name')

    # Check 3
    known = set(Schemes.__args__)
    tested = set(schemes.keys())
    if len(tested ^ known) != 0:
        pytest.fail(f'Smearing schemes known do not match those tested:\n'
                    f'\t- Schemes literal: {known}\n'
                    f'\t- Schemes tested: {tested}')


##################
# fermi_smearing #
##################
# Note that the single system test should be abstracted to a helper function.
def test_fermi_smearing_single(device):
    """Tests general & single system performance of `fermi_smearing`."""

    def ref(e, f, t):
        """Numpy based fermi smearing reference function."""
        return 1. / (np.exp((e - f) / t) + 1)

    # Check 1: ensure results are within tolerances. The `fermi_smearing`
    # function is rather simplistic, as such this check is mostly a sanity
    # check which ensures that the function does not return gibberish.
    eps_np = np.linspace(-5, 5, 100)
    eps = torch.tensor(eps_np, device=device)
    fe, kt = 0.5, 0.03
    pred = fermi_smearing(eps, torch.tensor(fe), kt)
    check_1 = np.allclose(pred.cpu().numpy(), ref(eps_np, fe, kt))

    assert check_1, 'Failed tolerance check'

    # Check 2: result should ∈ [0, 1].
    check_2 = fermi_smearing(torch.tensor(-1.), torch.tensor(0.), 1E-4) == 1
    assert check_2, 'Results not bound to the domain [0, 1]'

    # Check 3: an assertion error should be raised if `eigenvalues` &
    # `fermi_energy` are not tensors. This occurs when a smearing method does
    # not make a call to `_smearing_preprocessing` as expected.
    with pytest.raises(AssertionError):
        fermi_smearing(-1., torch.tensor(0.), 1E-4)

    with pytest.raises(AssertionError):
        fermi_smearing(torch.tensor(-0.), 0., 1E-4)

    # Check 4: device persistence check.
    check_4 = pred.device == device
    assert check_4, 'Device persistence check failed'

    # Check 5: ensure no "nan" values are encountered when kT is zero
    res = fermi_smearing(eps, torch.tensor(fe), 0.0)
    check_5 = torch.allclose(res, res)  # <- Note, "nan" ≠ "nan"
    assert check_5, '"nan" value found when kT=0.0'

    # Check 6: 0d tensors can be passed in
    try:
        pred_single = fermi_smearing(torch.tensor(1.0), torch.tensor(fe), kt)
        ref = ref(np.array(1.0), fe, kt)
        assert np.allclose(pred_single.cpu().numpy(), ref)
    except Exception as e:
        pytest.fail('Failed when given single eigenvalue')
        raise e


def test_fermi_smearing_batch(device):
    """Tests batch operability of the `fermi_smearing` function."""

    def ref(e, f, t):
        """Numpy based fermi smearing reference function."""
        return 1. / (np.exp((e - f) / t) + 1)

    eps_np = np.stack([np.linspace(-5, 5, 100), np.linspace(-3, 2, 100),
                       np.linspace(-1, 9, 100)])
    fe_np, kt_np = 0.5, 0.03
    kts_np = [0.001, 0.01, 0.1]
    fes_np = [0.5, 5.0, 10.0]

    eps = torch.tensor(eps_np, device=device)
    fe = torch.tensor(fe_np, device=device)
    kt = torch.tensor(kt_np, device=device)
    kts = torch.tensor(kts_np, device=device)
    fes = torch.tensor(fes_np, device=device)

    # Check 1: single global kT & Ef value.
    check_1 = np.allclose(
        fermi_smearing(eps, fe, kt).cpu().numpy(),
        np.stack([ref(i, fe_np, kt_np) for i in eps_np]))

    assert check_1, 'Failed on global-Ef/global-kT'

    # Check 2: single global kT but individual Ef values.
    check_2 = np.allclose(
        fermi_smearing(eps, fes, kt).cpu().numpy(),
        np.stack([ref(i, j, kt_np) for i, j in zip(eps_np, fes_np)]))

    assert check_2, 'Failed on individual-Ef/global-kT'

    # Check 3: single global Ef but individual kT values.
    check_3 = np.allclose(
        fermi_smearing(eps, fe, kts).cpu().numpy(),
        np.stack([ref(i, fe_np, j) for i, j in zip(eps_np, kts_np)]))

    assert check_3, 'Failed on global-Ef/individual-kT'

    # Check 4: individual kT & Ef values.
    check_4 = np.allclose(
        fermi_smearing(eps, fes, kts).cpu().numpy(),
        np.stack([ref(i, j, k) for i, j, k in zip(eps_np, fes_np, kts_np)]))

    assert check_4, 'Failed on individual-Ef/individual-kT'

    # Check 5: results do not change if kt/Ef values are in a more "batch-like"
    # shape, i.e. one entry per *row*.
    check_5 = torch.allclose(fermi_smearing(eps, fe, kts),
                             fermi_smearing(eps, fe.view(-1, 1), kts.view(-1, 1)))

    assert check_5, 'Failed when using reshaped Ef/kT tensors'

    # Check 6: device persistence check.
    check_6 = fermi_smearing(eps, fes, kts).device == device
    assert check_6, 'Device persistence check failed'


@pytest.mark.grad
def test_fermi_smearing_grad(device):
    """`fermi_smearing` gradient stability test."""
    molecules = [H2, H2O, CH4, H2COH, Au13]
    for mol in molecules:
        e_vals, kt, e_fermi, *_ = mol(device)
        e_vals.requires_grad = True
        check_1 = gradcheck(fermi_smearing, (e_vals, e_fermi['fermi'], kt),
                            raise_exception=False)
        assert check_1, f'Gradient check failed on {mol.__name__}'


#####################
# gaussian_smearing #
#####################
def test_gaussian_smearing_single(device):
    """Tests general & single system performance of `gaussian_smearing`."""

    def ref(e, f, t):
        """Numpy based gaussian smearing reference function."""
        return erfc((e - f) / t) / 2

    # Check 1: ensure results are within tolerances. The `gaussian_smearing`
    # function is rather simplistic, as such this check is mostly a sanity
    # check which ensures that the function does not return gibberish.
    eps_np = np.linspace(-5, 5, 100)
    eps = torch.tensor(eps_np, device=device)
    fe, kt = 0.5, 0.03
    pred = gaussian_smearing(eps, torch.tensor(fe), kt)
    check_1 = np.allclose(pred.cpu().numpy(), ref(eps_np, fe, kt))

    assert check_1, 'Failed tolerance check'

    # Check 2: result should be ∈ [0, 1].
    check_2 = gaussian_smearing(torch.tensor(-1.), torch.tensor(0.), 1E-4) == 1
    assert check_2, 'Results not bound to the domain [0, 1]'

    # Check 3: an assertion error should be raised if `eigenvalues` &
    # `fermi_energy` are not tensors. This occurs when a smearing method does
    # not make a call to `_smearing_preprocessing` as expected.
    with pytest.raises(AssertionError):
        gaussian_smearing(-1., torch.tensor(0.), 1E-4)

    with pytest.raises(AssertionError):
        gaussian_smearing(torch.tensor(-0.), 0., 1E-4)

    # Check 4: device persistence check.
    check_4 = pred.device == device
    assert check_4, 'Device persistence check failed'

    # Check 5: ensure no "nan" values are encountered when kT is zero
    res = gaussian_smearing(eps, torch.tensor(fe), 0.0)
    check_5 = torch.allclose(res, res)  # <- Note, "nan" ≠ "nan"
    assert check_5, '"nan" value found when kT=0.0'

    # Check 6: 0d tensors can be passed in
    try:
        pred_single = gaussian_smearing(torch.tensor(1.0), torch.tensor(fe), kt)
        ref = ref(np.array(1.0), fe, kt)
        assert np.allclose(pred_single.cpu().numpy(), ref)
    except Exception as e:
        pytest.fail('Failed when given single eigenvalue')
        raise e


def test_gaussian_smearing_batch(device):
    """Tests batch operability of the `gaussian_smearing` function."""

    def ref(e, f, t):
        """Numpy based gaussian smearing reference function."""
        return erfc((e - f) / t) / 2

    eps_np = np.stack([np.linspace(-5, 5, 100), np.linspace(-3, 2, 100),
                       np.linspace(-1, 9, 100)])
    fe_np, kt_np = 0.5, 0.03
    kts_np = [0.001, 0.01, 0.1]
    fes_np = [0.5, 5.0, 10.0]

    eps = torch.tensor(eps_np, device=device)
    fe = torch.tensor(fe_np, device=device)
    kt = torch.tensor(kt_np, device=device)
    kts = torch.tensor(kts_np, device=device)
    fes = torch.tensor(fes_np, device=device)

    # Check 1: single global kT & Ef value.
    check_1 = np.allclose(
        gaussian_smearing(eps, fe, kt).cpu().numpy(),
        np.stack([ref(i, fe_np, kt_np) for i in eps_np]))

    assert check_1, 'Failed on global-Ef/global-kT'

    # Check 2: single global kT but individual Ef values.
    check_2 = np.allclose(
        gaussian_smearing(eps, fes, kt).cpu().numpy(),
        np.stack([ref(i, j, kt_np) for i, j in zip(eps_np, fes_np)]))

    assert check_2, 'Failed on individual-Ef/global-kT'

    # Check 3: single global Ef but individual kT values.
    check_3 = np.allclose(
        gaussian_smearing(eps, fe, kts).cpu().numpy(),
        np.stack([ref(i, fe_np, j) for i, j in zip(eps_np, kts_np)]))

    assert check_3, 'Failed on global-Ef/individual-kT'

    # Check 4: individual kT & Ef values.
    check_4 = np.allclose(
        gaussian_smearing(eps, fes, kts).cpu().numpy(),
        np.stack([ref(i, j, k) for i, j, k in zip(eps_np, fes_np, kts_np)]))

    assert check_4, 'Failed on individual-Ef/individual-kT'

    # Check 5: results do not change if kt/Ef values are in a more "batch-like"
    # shape, i.e. one entry per *row*.
    check_5 = torch.allclose(gaussian_smearing(eps, fe, kts),
                             gaussian_smearing(eps, fe.view(-1, 1), kts.view(-1, 1)))

    assert check_5, 'Failed when using reshaped Ef/kT tensors'

    # Check 6: device persistence check.
    check_6 = gaussian_smearing(eps, fes, kts).device == device
    assert check_6, 'Device persistence check failed'


@pytest.mark.grad
def test_gaussian_smearing_grad(device):
    """`gaussian_smearing` gradient stability test."""
    molecules = [H2, H2O, CH4, H2COH, Au13]
    for mol in molecules:
        e_vals, kt, e_fermi, *_ = mol(device)
        e_vals.requires_grad = True
        check_1 = gradcheck(
            gaussian_smearing,(e_vals, e_fermi['gaussian'], kt),
            raise_exception=False)
        assert check_1, f'Gradient check failed on {mol.__name__}'


################
# fermi_search #
################
def test_fermi_search_general(device):
    """Test the basic functionality of the `fermi_search` function."""
    # This primarily ensures that the correct errors get raised.
    dv = {'device': device}
    ev = torch.tensor([0., 1., 2.], **dv)
    ev_batch = torch.tensor([[0., 1., 2.],
                             [0., 1., 2.]], **dv)
    mask = ev_batch.to(dtype=torch.bool)

    # Check 1: tolerance is too tight.
    with pytest.raises(ValueError, match='Tolerance*'):
        fermi_search(ev, 2, tolerance=1E-16)

    # Check 2: tolerance is negative
    with pytest.raises(ValueError, match='Tolerance value*'):
        fermi_search(ev, 2, tolerance=-1)

    # Check 3: kT is negative [single & batch]
    with pytest.raises(ValueError, match='kT must be*'):
        fermi_search(ev, 2, -0.1)

    with pytest.raises(ValueError, match='kT must be*'):
        fermi_search(ev_batch, torch.tensor([2, 2], **dv),
                     torch.tensor([1., -1.], **dv), e_mask=mask)

    # Check 4: no electrons [single & batch]
    with pytest.raises(ValueError, match='Number of elections cannot be zero'):
        fermi_search(ev, 0, 0.1)

    with pytest.raises(ValueError, match='Number of elections cannot be zero'):
        fermi_search(ev_batch, torch.tensor([2, 0], **dv),
                     torch.tensor([1., 1.], **dv), e_mask=mask)

    # Check 5: too many electrons [single & batch]
    with pytest.raises(ValueError, match='Number of electrons cannot exceed*'):
        fermi_search(ev, 10, 0.1)

    with pytest.raises(ValueError, match='Number of electrons cannot exceed*'):
        fermi_search(ev_batch, torch.tensor([2, 10], **dv),
                     torch.tensor([1., 1.], **dv), e_mask=mask)

    # Check 6: missing mask
    with pytest.raises(RuntimeError, match='A mask is required when in batch-mode!'):
        fermi_search(ev_batch, torch.tensor([2, 3], **dv),
                     torch.tensor([1., 2.], **dv))

    # Check 7: fermi search works without broadening (single and batch).
    fe_s = fermi_search(torch.tensor([0.0, 1.0], **dv), 2)
    fe_b = fermi_search(
        torch.tensor([[0.0, 1.0, 0.0], [1.0, 2.0, 3.0]], **dv),
        torch.tensor([2, 4], **dv))

    check_7a = torch.isclose(fe_s, torch.tensor(0.5, **dv))
    check_7b = torch.allclose(fe_b, torch.tensor([0.5, 2.5], **dv))

    assert check_7a, 'Fermi search failed without searing (single)'
    assert check_7b, 'Fermi search failed without searing (batch)'

    # Check 8: ensure a `basis` object can be used inplace of an e_mask
    e_vals, mask = pack([H2(device)[0], CH4(device)[0]], return_mask=True)
    basis = Basis(torch.tensor([[6, 1, 1, 1, 1],
                                [1, 1, 0, 0, 0]]), {1: [0], 6: [0, 1]})
    res_a = fermi_search(e_vals, 0.0036749, e_mask=mask)
    res_b = fermi_search(e_vals, 0.0036749, e_mask=basis)
    check_8 = torch.allclose(res_a, res_b)

    assert check_8, 'Failed to convert basis instance to mask.'


def test_fermi_search_single(device):
    """Check single system performance of the `fermi_search` function."""
    molecules = [H2, H2O, CH4, H2COH, Au13]

    # Ensure all test systems converge to the anticipated value. Both
    # fermi & gaussian schemes are tested here. Some molecules converge at the
    # middle-gap approximation while others require a full bisection search.
    for mol in molecules:
        e_vals, kt, e_fermi, _, n_elec = mol(device)
        ef_fermi = fermi_search(e_vals, n_elec, kt, scheme='fermi', max_iter=1000)
        ef_gaussian = fermi_search(e_vals, n_elec, kt, scheme='gaussian', max_iter=1000)

        # Check 1: tolerance check
        check_1a = torch.isclose(ef_fermi, e_fermi['fermi'])
        check_1b = torch.isclose(ef_gaussian, e_fermi['gaussian'])
        name = mol.__name__
        assert check_1a, f'{name} failed to converge with fermi smearing'
        assert check_1b, f'{name} failed to converge with gaussian smearing'

        # Check 2: Ensure masking works for single systems. If all state are
        # masked out correctly then there should be no place to put the
        # electrons, thus a "too many electrons" exception should be raised.
        with pytest.raises(ValueError, match='Number of electrons*'):
            _ = fermi_search(e_vals, n_elec, kt, scheme='fermi',
                             e_mask=torch.tensor(False, device=device))

        # Check 3: Ensure results are returned on the correct device
        check_3 = ef_fermi.device == device and ef_gaussian.device == device
        assert check_3, 'Device persistence check failed'


def test_fermi_search_batch(device):
    """Check batch system performance of the `fermi_search` function.

    Warnings:
        This function is dependant on `torch.common.batch.pack`.
    """

    # Ensure all test systems converge to the anticipated value. Both
    # fermi & gaussian schemes are tested here. Some molecules converge at the
    # middle-gap approximation while others require a full bisection search.
    mols = [i(device) for i in [H2, H2O, CH4, H2COH, Au13]]
    e_vals, mask = pack([i[0] for i in mols], return_mask=True)
    kt = torch.stack([i[1] for i in mols])
    n_elec = torch.tensor([i[4] for i in mols], device=device)

    ef_fermi = fermi_search(e_vals, n_elec, kt, scheme='fermi', e_mask=mask)
    ef_gaussian = fermi_search(e_vals, n_elec, kt, scheme='gaussian', e_mask=mask)

    # Check 1: tolerance check
    check_1a = torch.allclose(ef_fermi, torch.stack([i[2]['fermi'] for i in mols]))
    check_1b = torch.allclose(ef_gaussian, torch.stack([i[2]['gaussian'] for i in mols]))

    assert check_1a, 'Failed to converge with fermi smearing'
    assert check_1b, 'Failed to converge with gaussian smearing'

    # Check 2: Ensure results are returned on the correct device
    check_2 = ef_fermi.device == device and ef_gaussian.device == device
    assert check_2, 'Device persistence check failed'


##############
# ft_entropy #
##############
def test_smearing_entropy_single(device):
    """Test single system performance of the `ft_entropy` function."""

    molecules = [H2, H2O, CH4, H2COH, Au13]
    for mol in molecules:
        #
        e_vals, kt, e_fermi, entropy, _ = mol(device)
        ts_fermi = smearing_entropy(e_vals, e_fermi['fermi'], kt, 'fermi')
        ts_gaussian = smearing_entropy(e_vals, e_fermi['gaussian'], kt, 'gaussian')

        # Check 2: Tolerance check
        check_1a = torch.isclose(ts_fermi, entropy['fermi'])
        check_1b = torch.isclose(ts_gaussian, entropy['gaussian'])
        name = mol.__name__
        assert check_1a, f'Incorrect fermi-entropy value returned for {name}'
        assert check_1b, f'Incorrect gaussian-entropy value returned for {name}'

        # Check 2: Ensure masking works for single systems by masking all states.
        ts = smearing_entropy(e_vals, e_fermi['fermi'], kt, 'fermi', torch.tensor(False))
        check_2 = torch.allclose(ts, torch.tensor(0., device=device))
        assert check_2, 'Attempt to mask states failed'

        # Check 3: Ensure results are returned on the correct device
        check_3 = ts_fermi.device == device and ts_gaussian.device == device
        assert check_3, 'Device persistence check failed'


def test_smearing_entropy_batch(device):
    """Test batch system performance of the `ft_entropy` function.

    Warnings:
        This function is dependant on `torch.common.batch.pack`.
    """

    mols = [i(device) for i in [H2, H2O, CH4, H2COH, Au13]]
    e_vals, mask = pack([i[0] for i in mols], return_mask=True)
    kt = torch.stack([i[1] for i in mols])
    ef_fermi = torch.stack([i[2]['fermi'] for i in mols])
    ef_gaussian = torch.stack([i[2]['gaussian'] for i in mols])

    ts_fermi= smearing_entropy(e_vals, ef_fermi, kt, 'fermi', mask)
    ts_gaussian = smearing_entropy(e_vals, ef_gaussian, kt, 'gaussian', mask)

    # Check 1: tolerance check
    check_1a = torch.allclose(ts_fermi,
                              torch.stack([i[3]['fermi'] for i in mols]))
    check_1b = torch.allclose(ts_gaussian,
                              torch.stack([i[3]['gaussian'] for i in mols]))

    assert check_1a, 'Incorrect fermi-entropy value returned'
    assert check_1b, 'Incorrect gaussian-entropy value returned'

    # Check 2: ensure an error is raised when passing in a batch without a mask.
    with pytest.raises(RuntimeError, match='A mask is required when*'):
        smearing_entropy(e_vals, ef_fermi, kt, 'fermi')

    # Check 3: Ensure results are returned on the correct device
    check_3 = ts_fermi.device == device and ts_gaussian.device == device
    assert check_3, 'Device persistence check failed'


    # Check 4: ensure a `basis` object can be used inplace of an e_mask
    basis = Basis(
        [torch.tensor(i) for i in [[1, 1], [8, 1, 1], [6, 1, 1, 1, 1], [1, 1, 1, 6, 8], [79] * 13]],
        {1: [0], 6: [0, 1], 8: [0, 1], 79: [0, 1, 2]})

    res_a = smearing_entropy(e_vals, ef_fermi, 0.0036749, 'fermi', e_mask=mask)
    res_b = smearing_entropy(e_vals, ef_fermi, 0.0036749, 'fermi', e_mask=basis)

    check_4 = torch.allclose(res_a, res_b)

    assert check_4, 'Failed to convert basis instance to mask.'


@pytest.mark.grad
def test_smearing_entropy_grad(device):
    """Test the gradient stability of the `ft_entropy` method."""

    molecules = [H2, H2O, CH4, H2COH, Au13]
    for mol in molecules:
        e_vals, kt, e_fermi, *_ = mol(device)
        e_vals.requires_grad = True
        check_1a = gradcheck(
            smearing_entropy, (e_vals, e_fermi['fermi'], kt, 'fermi'),
            raise_exception=False)
        check_1b = gradcheck(
            smearing_entropy, (e_vals, e_fermi['gaussian'], kt, 'gaussian'),
            raise_exception=False)

        assert check_1a, f'Fermi based gradient check failed on {mol.__name__}'
        assert check_1b, f'Gaussian based gradient check failed on {mol.__name__}'
