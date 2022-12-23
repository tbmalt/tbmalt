from torch.autograd import gradcheck
import pytest
from tbmalt.physics.dftb.properties import dos, band_pass_state_filter
from tests.test_utils import *
from tests.unittests.data.properties.dos import H2, CH4, HCOOH
from scipy.signal import find_peaks, peak_widths
from tbmalt.common.batch import pack



####################
# Helper Functions #
####################
def peak_id(x, y):
    positions_index = find_peaks(y, 1E-11)[0]
    heights = y[positions_index]
    widths = np.hstack(peak_widths(y, positions_index))
    return x[positions_index], heights, widths


def compare_distributions(energies, dist_a, dist_b):
    """Compares a pair of distributions.

    The method by which a distribution can be constructed depends heavily on
    how broadening is implemented. As such an implementation varies from code
    to code it is better to compare the peaks rather than the distribution as
    a whole.
    """
    # Must move to cpu and convert to numpy.
    pos_a, height_a, width_a = peak_id(energies.sft(), dist_a.sft())
    pos_b, height_b, width_b = peak_id(energies.sft(), dist_b.sft())
    are_close = all([
        np.allclose(pos_a, pos_b),
        np.allclose(height_a, height_b),
        np.allclose(width_a, width_b)])
    return are_close


def get_data(dataset, device, grad=False):
    """Fetches reference data for P/DoS calculations.

    Arguments:
        dataset: The test dataset to load.
        device: the device on which the data should be placed.
        grad: Is data to be used in a `gradcheck`. If "True", i) requires_grad
            is enabled where required, & ii) the energy tensor is truncated to
            reduce computational overhead.

    Returns:
        data: A tuple containing:
            (eigenvalues, smearing_width, fermi_energy, energies,
             reference_dos, reference_pdos, bases, eigenvectors,
             overlap_matrix)
    Notes:
        It is assume that, for a given system, the DoS and PDoS were
        calculated over the same range.

    Warnings:
        Tests should never the returned arrays as they are hardlinked. Meaning
        that all subsequent tests will use the modified arrays, and will thus
        fail.

    """
    eps = dataset['eigenvalues'].clone().to(device)
    sigma = dataset['sigma']
    fermi = dataset['fermi']
    energies = dataset['dos']['energy'].clone().to(device)
    ref_dos = dataset['dos']['total']
    ref_pdos = dataset['pdos']
    bases = dataset['bases']
    C = dataset['eigenvectors'].clone().to(device)
    S = dataset['overlap'].clone().to(device)

    if grad:
        energies = energies[::10]
        for i in [eps, S, C]:
            i.requires_grad = True

    return eps, sigma, fermi, energies, ref_dos, ref_pdos, bases, C, S


####################################################
# TBMaLT.physics.properties.band_pass_state_filter #
####################################################
def test_band_pass_state_filter_single(device):
    """Tests single-system performance of the band_pass_state_filter function.

    Single-system tests performed on the band_pass_state_filter function:
        1. Ensure result is within tolerance.
        2. Zero padded packing does not interfere with filter operation.
        3. Mask is returned on the correct device.

    Notes:

        Gradient tests are not performed on this function as gradients are not
        calculated through it; and are thus not applicable.
    """
    # Check 1: the expected result is returned under standard conditions.
    a = torch.linspace(-4, 4, 9, device=device)
    mask = band_pass_state_filter(a, 3, 2, 0.0)
    ref_mask = torch.tensor([0, 0, 1, 1, 1, 1, 1, 0, 0],
                            dtype=torch.bool, device=device)
    chk_1 = (mask == ref_mask).all()

    assert chk_1, 'Incorrect mask returned.'

    # Check 2: that zero padded packing does not break the mask function.
    a[-1:2] = 0
    mask = band_pass_state_filter(a, 3, 2, 0.0)
    chk_2 = (mask == ref_mask).all()

    assert chk_2, 'Zero-padded packing result in incorrect mask generation'

    # Check 3: that the mask is on the correct device
    chk_3 = mask.device == device

    assert chk_3, 'Mask was generated on the wrong device.'


def test_band_pass_state_filter_batch(device):
    """Tests batch-system performance of the band_pass_state_filter function.

    Batch-system tests performed on the band_pass_state_filter function:
        1. Ensure result is within tolerance.
        2. Zero padded packing does not interfere with filter operation.
        3. Mask is returned on the correct device.
        4. Exceptions raised if n_homo, n_lumo, & fermi values are not
            specified for each system

    """
    a = torch.stack((
        torch.linspace(-4, 5, 10, device=device),
        torch.linspace(-3, 6, 10, device=device),
        torch.linspace(-2, 7, 10, device=device),
    ))

    ref_mask = torch.tensor([
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    ], dtype=torch.bool, device=device)

    n = torch.tensor([2, 2, 2], device=device)
    fermi = torch.tensor([0., 0., 0.], device=device)

    # Check 1: the expected result is returned under standard conditions.
    chk_1 = (band_pass_state_filter(a, n, n, fermi) == ref_mask).all()

    assert chk_1, 'Incorrect mask returned.'

    # Check 2: that zero padded packing does not break the mask function.
    a[:, -1] = 0

    chk_2 = (band_pass_state_filter(a, n, n, fermi) == ref_mask).all()

    assert chk_2, 'Zero-padded packing result in incorrect mask generation'

    # Check 3: that the mask is on the correct device
    chk_3 = band_pass_state_filter(a, n, n, fermi).device == device

    assert chk_3, 'Mask was generated on the wrong device.'

    # Check 4: an exception is raised if they user does not pass a n_homo,
    # n_lumo, & fermi argument for each system.
    with pytest.raises(RuntimeError):
        band_pass_state_filter(a, 1, n, fermi)
    with pytest.raises(RuntimeError):
        band_pass_state_filter(a, n, 1, fermi)
    with pytest.raises(RuntimeError):
        band_pass_state_filter(a, n, n, 1)


#################################
# TBMaLT.physics.properties.dos #
#################################
def test_dos_single(device):
    """Tests single-system operability of the dos function.

    This tests the single-system operability of the density of states function
    "dos". Tests are performed to ensure that:
        1. Resulting distributions are within acceptable tolerances.
        2. Offset operation works as intended.
        3. Masking produces the expected result.
        4. The scale option rescales the DoS to the domain [0,1].
        5. Result is returned on the correct device.

    Notes:
        This test uses data produced by FHI-aims to construct a density of
        states distribution. This distribution is then compared to that
        produced by FHI-aims.

        It should be noted that currently a deviation, of less than 1E-5, is
        encountered when comparing the FHI-aims computed DoS to that produced
        by TBMaLT's dos function. The source of this error is currently
        unknown, however, it is thought to emanate form differences in unit
        conversions.

    """
    # Load the test data for CH4
    eps, sigma, fermi, energies, ref_dos, *_ = get_data(CH4, device)

    # Check 1: Ensure predicted & reference DoSs are within tolerance
    # thresholds.
    pred_dos = dos(eps, energies, sigma)
    chk_1 = compare_distributions(energies, pred_dos, ref_dos)

    assert chk_1, 'Calculated DoS does not match the reference value.'

    # Check 2: Offset should shift the DoS so that fermi becomes the zero
    # point.
    pred_dos_offset = dos(eps, energies - fermi, sigma, fermi)
    chk_2 = np.allclose(find_peaks(ref_dos.sft(), 1E-11)[0],
                        find_peaks(pred_dos_offset.sft(), 1E-11)[0])
    assert chk_2, 'Offset operation was not completed correctly.'

    # Check 3: Ensure masking is working correctly.
    # If running this test on another system an issue may be encountered where
    # two peaks overlap and thus get an incorrect number of peaks counted.
    mask = band_pass_state_filter(eps, 1, 1, fermi)
    dos_b = dos(eps, energies, sigma, mask=mask)
    peaks_b = energies[find_peaks(dos_b.sft(), 1E-11)[0]]
    chk_3 = all([
        # Masked DoS should have only 2 states
        len(peaks_b) == 2,
        # There should be 1 LUMO state after masking
        len(np.where(peaks_b.sft() > fermi)[0]) == 1,
        # There should be 1 HOMO state after masking
        len(np.where(peaks_b.sft() <= fermi)[0]) == 1,
    ])

    assert chk_3, 'Masking did not proceed as anticipated.'

    # Check 4: Confirm that scaling does, in-fact, work.
    chk_4 = torch.allclose(dos(eps, energies, sigma, scale=True).max(),
                           torch.tensor(1.0, device=device))

    assert chk_4, 'Scaling failed to map DoS to the domain of [0, 1].'

    # Check 5: Perform a device persistancy check.
    chk_5 = pred_dos.device == device

    assert chk_5, 'DoS was placed on the wrong device.'


def test_dos_batch(device):
    """Tests batch-system operability of the dos function.

    This tests the batch-system operability of the density of states function
    "dos". Tests are performed to ensure that:
        1. Resulting distributions are within acceptable tolerances.
        2. Offset operation works as intended.
        3. Masking produces the expected result.
        4. The scale option rescales the DoS to the domain [0,1].
        5. Result is returned on the correct device.

    """
    # Load in the test data; H2, CH4, & HCOOH.
    eps_1, sigma_1, fermi_1, energies_1, ref_dos_1, *_ = get_data(H2, device)
    eps_2, sigma_2, fermi_2, energies_2, ref_dos_2, *_ = get_data(CH4, device)
    eps_3, sigma_3, fermi_3, energies_3, ref_dos_3, *_ = get_data(HCOOH, device)

    # Pack the data together, making sure to get the eps mask.
    eps, eps_mask = pack([eps_1, eps_2, eps_3], return_mask=True)
    energies = pack([energies_1, energies_2, energies_3])
    sigma = torch.tensor([sigma_1, sigma_2, sigma_3], device=device)
    fermi = torch.tensor([fermi_1, fermi_2, fermi_3], device=device)
    ref_dos = pack([ref_dos_1, ref_dos_2, ref_dos_3])

    # Check 1: Ensure predicted DoSs are within tolerance thresholds.
    pred_dos = dos(eps, energies, sigma, mask=eps_mask)
    chk_1 = all([compare_distributions(energies_1, pred_dos[0], ref_dos_1),
                 compare_distributions(energies_2, pred_dos[1], ref_dos_2),
                 compare_distributions(energies_3, pred_dos[2], ref_dos_3)])

    assert chk_1, 'Calculated DoS does match not the reference value.'

    # Check 2: Offset should shift the DoS so that fermi becomes the 0 point.
    pred_dos_offset = dos(eps, energies - fermi[:, None], sigma, fermi,
                          mask=eps_mask)

    chk_2 = all([
        np.allclose(
            find_peaks(ref_dos[0].sft(), 1E-11)[0],
            find_peaks(pred_dos_offset.sft()[0], 1E-11)[0])
        for i in range(ref_dos.shape[0])])

    assert chk_2, 'Offset operation was not completed correctly.'

    # Check 3: Ensure masking is working correctly.
    # If running this test on another system an issue may be encountered where
    # two peaks overlap and thus get an incorrect number of peaks counted.
    mask = band_pass_state_filter(eps, torch.tensor([1, 1, 1], device=device),
                                  torch.tensor([1, 1, 1], device=device),
                                  fermi) & eps_mask

    dos_b = dos(eps, energies, sigma, mask=mask)
    peaks_b_1 = energies[0][find_peaks(dos_b[0].sft(), 1E-11)[0]]
    peaks_b_2 = energies[1][find_peaks(dos_b[1].sft(), 1E-11)[0]]

    # System 3 is ignored as it is not necessary.
    chk_3 = all([
        len(peaks_b_1) == 2,
        len(torch.where(peaks_b_1 > fermi[0])[0]) == 1,
        len(torch.where(peaks_b_1 <= fermi[0])[0]) == 1,
        len(peaks_b_2) == 2,
        len(torch.where(peaks_b_2 > fermi[1])[0]) == 1,
        len(torch.where(peaks_b_2 <= fermi[1])[0]) == 1,
    ])

    assert chk_3, 'Masking did not proceed as anticipated.'

    # Check 4: Confirm that scaling does, in-fact, work.
    dos_c = dos(eps, energies, sigma, scale=True)
    max_vals = torch.stack((dos_c[0].max(), dos_c[1].max(), dos_c[2].max()))
    chk_4 = torch.allclose(max_vals, torch.tensor(1.0, device=device))

    assert chk_4, 'Scaling failed to map DoS to the domain of [0, 1].'

    # Check 5: Perform a device persistancy check.
    chk_5 = pred_dos.device == device

    assert chk_5, 'DoS was placed on the wrong device.'


@pytest.mark.grad
def test_dos_grad(device):
    """Tests gradient stability of the dos function.

    This test ensures that the gradient can be reliably calculated through
    the dos function for both a single system and a batch of systems.

    Notes:
        As this test is computational expensive; the energies at which the dos
        dos is calculated has been restricted. Only two tests are performed,
        one for a single system and one for a batch of systems. The `offset`,
        `scale`, & `mask` are not tested separately due to cost of doing so.

    """
    # Load in the test data; H2 & CH4
    eps_1, sigma_1, fermi_1, energies_1, *_ = get_data(H2, device, True)
    eps_2, sigma_2, fermi_2, energies_2, *_ = get_data(CH4, device, True)

    # Test gradient for single system
    eps_mask = band_pass_state_filter(eps_2, 1, 1, fermi_2)
    chk_1 = gradcheck(dos, (eps_2, energies_2, sigma_2, fermi_2, eps_mask, True),
                      raise_exception=False)
    assert chk_1, 'dos function failed gradient stability check (single).'

    # Test gradient for a batch of systems
    eps, eps_mask = pack([eps_1, eps_2], return_mask=True)
    energies = pack([energies_1, energies_2])
    sigma = torch.tensor([sigma_1, sigma_2], device=device)
    fermi = torch.tensor([fermi_1, fermi_2], device=device,
                         requires_grad=True)

    chk_2 = gradcheck(dos, (eps, energies, sigma, fermi, eps_mask, True),
                      raise_exception=False)
    assert chk_2, 'dos function failed gradient stability check (batch).'
