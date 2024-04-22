import pytest

import torch
from torch.autograd import gradcheck
import torch.nn as nn
from tests.test_utils import fix_seed

from tbmalt import Geometry, OrbitalInfo
from tbmalt.physics.dftb import Dftb2
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed
from tbmalt.ml.loss_function import Loss
from tbmalt.ml.loss_function import l1_loss, mse_loss, hellinger_loss
from tbmalt.common.batch import pack

from tests.test_utils import skf_file

torch.set_default_dtype(torch.float64)


##################
# Loss test data #
##################
@fix_seed
def loss_prediction_data(device, batch=False, requires_grad=False):
    rg = requires_grad
    if not batch:
        return torch.rand(5, device=device, requires_grad=rg)
    else:
        return [torch.rand(5, device=device, requires_grad=rg),
                torch.rand(3, device=device, requires_grad=rg),
                torch.rand(2, device=device, requires_grad=rg)]


def loss_reference_data(device, batch=False, requires_grad=False):
    rg = requires_grad
    if not batch:
        return torch.rand(5, device=device, requires_grad=rg)
    else:
        return [torch.rand(5, device=device, requires_grad=rg),
                torch.rand(3, device=device, requires_grad=rg),
                torch.rand(2, device=device, requires_grad=rg)]


#################
# Loss function #
#################
def loss_function_helper(prediction, reference, device):
    """Function to reduce code duplication when testing loss functions."""
    batch = isinstance(prediction, list) or prediction.ndim == 2
    rg = prediction.requires_grad if not batch else prediction[0].requires_grad
    weights = [torch.tensor([0.2], device=device), torch.tensor([0.3], device=device),
               torch.tensor([0.5], device=device)]
    weights = pack(weights)

    if batch:
        prediction = pack(prediction)
        reference = pack(reference)

    if not batch:
        # Check 1: Ensure loss functions return correct results
        # Check 1a: Single system without weights
        # l1 loss
        loss_l1_tbmalt = l1_loss(prediction, reference)
        loss_func_l1 = nn.L1Loss()
        loss_l1_ref = loss_func_l1(prediction, reference)
        check_1a_1 = torch.allclose(loss_l1_tbmalt, loss_l1_ref)
        assert check_1a_1, 'l1 loss (single system) does not give the correct value.'

        # MSE loss
        loss_mse_tbmalt = mse_loss(prediction, reference)
        loss_func_mse = nn.MSELoss()
        loss_mse_ref = loss_func_mse(prediction, reference)
        check_1a_2 = torch.allclose(loss_mse_tbmalt, loss_mse_ref)
        assert check_1a_2, 'mse loss (single system) does not give the correct value.'

    if batch:
        # Check 1b: Batch systems without weights
        # l1 loss
        loss_l1_tbmalt_b = l1_loss(prediction, reference)
        loss_func_l1_b = nn.L1Loss()
        loss_l1_ref_b = loss_func_l1_b(prediction, reference)
        check_1b_1 = torch.allclose(loss_l1_tbmalt_b, loss_l1_ref_b)
        assert check_1b_1, 'l1 loss (batch systems) does not give the correct value.'

        # MSE loss
        loss_mse_tbmalt_b = mse_loss(prediction, reference)
        loss_func_mse_b = nn.MSELoss()
        loss_mse_ref_b = loss_func_mse_b(prediction, reference)
        check_1b_2 = torch.allclose(loss_mse_tbmalt_b, loss_mse_ref_b)
        assert check_1b_2, 'mse loss (batch systems) does not give the correct value.'

        # Hellinger loss
        loss_hellinger_tbmalt_b = hellinger_loss(prediction, reference)
        loss_hellinger_ref_b = torch.tensor([hellinger_loss(prediction[0], reference[0]),
                                             hellinger_loss(prediction[1], reference[1]),
                                             hellinger_loss(prediction[2], reference[2])],
                                            device=device).mean()
        check_1c_3 = torch.allclose(loss_hellinger_tbmalt_b, loss_hellinger_ref_b)
        assert check_1c_3, 'Hellinger loss (batch systems) does not give the correct value.'

        # Check 1c: Batch systems with weights
        # l1 loss
        loss_l1_tbmalt_w = l1_loss(prediction, reference, weights)
        loss_func_l1_w = nn.L1Loss(reduction='none')
        loss_l1_ref_w = (loss_func_l1_w(prediction, reference) * weights).mean()
        check_1c_1 = torch.allclose(loss_l1_tbmalt_w, loss_l1_ref_w)
        assert check_1c_1, 'l1 loss with weights does not give the correct value.'

        # MSE loss
        loss_mse_tbmalt_w = mse_loss(prediction, reference, weights)
        loss_func_mse_w = nn.MSELoss(reduction='none')
        loss_mse_ref_w = (loss_func_mse_w(prediction, reference) * weights).mean()
        check_1c_2 = torch.allclose(loss_mse_tbmalt_w, loss_mse_ref_w)
        assert check_1c_2, 'mse loss with weights does not give the correct value.'

        # Hellinger loss
        loss_hellinger_tbmalt_w = hellinger_loss(prediction, reference, weights)
        loss_hellinger_ref_w = torch.tensor([hellinger_loss(prediction[0], reference[0]) * 0.2,
                                             hellinger_loss(prediction[1], reference[1]) * 0.3,
                                             hellinger_loss(prediction[2], reference[2]) * 0.5],
                                            device=device).mean()
        check_1c_3 = torch.allclose(loss_hellinger_tbmalt_w, loss_hellinger_ref_w)
        assert check_1c_3, 'Hellinger loss with weights does not give the correct value.'

        # Check 1d: Batch systems with weights using 'sum' reduction method
        # l1 loss
        loss_l1_tbmalt_w_s = l1_loss(prediction, reference, weights, 'sum')
        loss_func_l1_w = nn.L1Loss(reduction='none')
        loss_l1_ref_w_s = (loss_func_l1_w(prediction, reference) * weights).sum()
        check_1d_1 = torch.allclose(loss_l1_tbmalt_w_s, loss_l1_ref_w_s)
        assert check_1d_1, 'l1 loss with weights using sum reduction does not give the correct value.'

        # MSE loss
        loss_mse_tbmalt_w_s = mse_loss(prediction, reference, weights, 'sum')
        loss_func_mse_w = nn.MSELoss(reduction='none')
        loss_mse_ref_w_s = (loss_func_mse_w(prediction, reference) * weights).sum()
        check_1d_2 = torch.allclose(loss_mse_tbmalt_w_s, loss_mse_ref_w_s)
        assert check_1d_2, 'mse loss with weights using sum reduction does not give the correct value.'

        # Hellinger loss
        loss_hellinger_tbmalt_w_s = hellinger_loss(prediction, reference, weights, 'sum')
        loss_hellinger_ref_w_s = torch.tensor([hellinger_loss(prediction[0], reference[0]) * 0.2,
                                               hellinger_loss(prediction[1], reference[1]) * 0.3,
                                               hellinger_loss(prediction[2], reference[2]) * 0.5],
                                              device=device).sum()
        check_1d_3 = torch.allclose(loss_hellinger_tbmalt_w_s, loss_hellinger_ref_w_s)
        assert check_1d_3, 'Hellinger loss with weights using sum reduction does not give the correct value.'


    # Check 2: Gradient check
    if rg:
        if not batch:
            # Check 2a: Single system without weights
            # l1 loss
            check_2a_1 = gradcheck(l1_loss, (prediction, reference),
                                     raise_exception=False)
            assert check_2a_1, 'l1 loss (single system) does not pass the gradient check.'

            # MSE loss
            check_2a_2 = gradcheck(mse_loss, (prediction, reference),
                                     raise_exception=False)
            assert check_2a_2, 'mse loss (single system) does not pass the gradient check.'

            # Hellinger loss
            check_2a_3 = gradcheck(hellinger_loss, (prediction, reference),
                                     raise_exception=False)
            assert check_2a_3, 'hellinger loss (single system) does not pass the gradient check.'

        if batch:
            # Check 2b: Batch systems without weights
            # l1 loss
            check_2b_1 = gradcheck(l1_loss, (prediction, reference),
                                     raise_exception=False)
            assert check_2b_1, 'l1 loss (batch systems) does not pass the gradient check.'

            # MSE loss
            check_2b_2 = gradcheck(mse_loss, (prediction, reference),
                                     raise_exception=False)
            assert check_2b_2, 'mse loss (batch systems) does not pass the gradient check.'

            # Check 2c: Batch systems with weights
            # l1 loss
            check_2c_1 = gradcheck(l1_loss, (prediction, reference, weights),
                                     raise_exception=False)
            assert check_2c_1, 'l1 loss with weights does not pass the gradient check.'

            # MSE loss
            check_2c_2 = gradcheck(mse_loss, (prediction, reference, weights),
                                     raise_exception=False)
            assert check_2c_2, 'mse loss with weights does not pass the gradient check.'


def test_loss_test_single(device):
    """Test the functionality of the loss class for a single system."""
    loss_function_helper(loss_prediction_data(device), loss_reference_data(device),
                         device)
    loss_function_helper(loss_prediction_data(device, requires_grad=True),
                         loss_reference_data(device, requires_grad=True),
                         device)


def test_loss_test_batch(device):
    """Test the functionality of the loss class for batch systems."""
    loss_function_helper(loss_prediction_data(device, batch=True),
                         loss_reference_data(device, batch=True),
                         device)
    loss_function_helper(loss_prediction_data(device, batch=True, requires_grad=True),
                         loss_reference_data(device, batch=True, requires_grad=True),
                         device)


########################
# Dataset for unittest #
########################
def data_delegate(device):
    """A dataset for unittest."""
    numbers = [torch.tensor([1, 1], device=device),
              torch.tensor([8, 1, 1], device=device),
              torch.tensor([6, 1, 1, 1, 1], device=device)]
    positions = [torch.tensor([[ 0.0000,  0.0000,  0.6965],
                               [ 0.0000,  0.0000, -0.6965]], device=device),
                 torch.tensor([[ 0.0000,  0.0000,  0.2254],
                               [ 0.0000,  1.4423, -0.9015],
                               [ 0.0000, -1.4423, -0.9015]], device=device),
                 torch.tensor([[ 0.0000,  0.0000,  0.0000],
                               [ 1.1889,  1.1889,  1.1889],
                               [-1.1889, -1.1889,  1.1889],
                               [ 1.1889, -1.1889, -1.1889],
                               [-1.1889,  1.1889, -1.1889]], device=device)]
    q_ref = [torch.tensor([1.0, 1.0], device=device),
             torch.tensor([6.0, 1.0, 1.0], device=device),
             torch.tensor([4.0, 1.0, 1.0, 1.0, 1.0], device=device)]
    fermi_ref = [torch.tensor(-0.25, device=device),
                 torch.tensor(-0.20, device=device),
                 torch.tensor(-0.24, device=device)]
    system_weight = [torch.tensor([0.2], device=device),
                     torch.tensor([0.3], device=device),
                     torch.tensor([0.5], device=device)]
    dataset = {"atomic_numbers": numbers, "positions": positions,
              "q_final_atomic": pack(q_ref), "fermi_energy": pack(fermi_ref),
              "system_weight": pack(system_weight)}

    return dataset


#############
# Delegates #
#############
def prediction_delegate_mismatch(calculator, dataset, **kwargs):
    predictions = dict()
    predictions["q_final_atomic"] = calculator.q_final_atomic
    return predictions


def reference_delegate_mismatch(calculator, dataset, **kwargs):
    references = dict()
    references["q_final_atomic"] = dataset["q_final_atomic"]
    references["fermi_energy"] = dataset["fermi_energy"].unsqueeze(-1)
    return references


def prediction_delegate(calculator, dataset, **kwargs):
    predictions = dict()
    predictions["q_final_atomic"] = calculator.q_final_atomic
    predictions["fermi_energy"] = calculator.fermi_energy
    return predictions


def reference_delegate(calculator, dataset, **kwargs):
    references = dict()
    references["q_final_atomic"] = dataset["q_final_atomic"]
    references["fermi_energy"] = dataset["fermi_energy"]
    return references


def system_weight_delegate(calculator, dataset):
    system_weight = dataset["system_weight"]
    return system_weight


##############################################
# Compatibility of loss class and calculator #
##############################################
@pytest.fixture
def scc_feeds(device, skf_file):
    """Feeds for DFTB2."""
    species = [1, 6, 8]
    h_feed = SkFeed.from_database(skf_file, species, 'hamiltonian', device=device)
    s_feed = SkFeed.from_database(skf_file, species, 'overlap', device=device)
    o_feed = SkfOccupationFeed.from_database(skf_file, species, device=device)
    u_feed = HubbardFeed.from_database(skf_file, species, device=device)

    return h_feed, s_feed, o_feed, u_feed


def loss_class_helper(scc_feeds, data_delegate, prediction_delegate, reference_delegate, device):
    """Function to reduce code duplication when testing loss class."""
    numbers = data_delegate["atomic_numbers"]
    positions = data_delegate["positions"]
    shell_dict = {1: [0], 6: [0, 1], 8:[0, 1]}
    geometry = Geometry(numbers, positions, units='a')
    orbs = OrbitalInfo(geometry.atomic_numbers, shell_dict, shell_resolved=False)
    h_feed, s_feed, o_feed, u_feed = scc_feeds
    kwargs = {'filling_scheme': 'fermi', 'filling_temp': 0.0036749324}
    dftb_calculator = Dftb2(h_feed, s_feed, o_feed, u_feed, **kwargs)
    dftb_calculator(geometry, orbs)

    # Check 1: Warn should be given if the keys in prediction & reference mismatch
    loss_entity_1 = Loss(prediction_delegate_mismatch, reference_delegate)
    with pytest.warns(UserWarning, match=r'Missmatch detected*'):
        _ = loss_entity_1(dftb_calculator, data_delegate)

    # Check 2: Error should be raised if the prediction & reference size mismatch
    loss_entity_2 = Loss(prediction_delegate, reference_delegate_mismatch)
    with pytest.raises(AssertionError, match=r'prediction & reference size mismatch'):
        _ = loss_entity_2(dftb_calculator, data_delegate)

    # Check 3: Check the results of raw and total loss
    # Check 3a: l1 loss without weights for different properties
    loss_entity_3a = Loss(prediction_delegate, reference_delegate)
    total_loss_3a, raw_losses_3a = loss_entity_3a(dftb_calculator, data_delegate)
    loss_func_l1_ref = nn.L1Loss()
    loss_q_ref_3a = loss_func_l1_ref(dftb_calculator.q_final_atomic,
                                  data_delegate["q_final_atomic"])
    loss_fermi_ref_3a = loss_func_l1_ref(dftb_calculator.fermi_energy,
                                         data_delegate["fermi_energy"])
    check_3a_1 = torch.allclose(raw_losses_3a["q_final_atomic"], loss_q_ref_3a)
    check_3a_2 = torch.allclose(raw_losses_3a["fermi_energy"], loss_fermi_ref_3a)
    check_3a_3 = torch.allclose(total_loss_3a, loss_q_ref_3a + loss_fermi_ref_3a)
    check_3a = check_3a_1 and check_3a_2 and check_3a_3

    assert check_3a, 'l1 loss without weights for different properties does not give the correct value.'

    # Check 3b: different loss functions without weights for different properties
    loss_entity_3b = Loss(prediction_delegate, reference_delegate,
                          loss_functions={"q_final_atomic": l1_loss,
                                          "fermi_energy": mse_loss})
    total_loss_3b, raw_losses_3b = loss_entity_3b(dftb_calculator, data_delegate)
    loss_func_mse_ref = nn.MSELoss()
    loss_fermi_ref_3b = loss_func_mse_ref(dftb_calculator.fermi_energy,
                                          data_delegate["fermi_energy"])
    check_3b_1 = torch.allclose(raw_losses_3b["fermi_energy"], loss_fermi_ref_3b)
    check_3b_2 = torch.allclose(total_loss_3b, loss_q_ref_3a + loss_fermi_ref_3b)
    check_3b = check_3b_1 and check_3b_2

    assert check_3b, ('Loss calculated by different loss functions without weights '
                     'for different properties does not give the correct value.')

    # Check 3c: different loss functions with weights for different properties
    loss_entity_3c = Loss(prediction_delegate, reference_delegate,
                          loss_functions={"q_final_atomic": l1_loss,
                                          "fermi_energy": mse_loss},
                          loss_weights={"q_final_atomic": 0.7,
                                        "fermi_energy": 0.3})
    total_loss_3c, raw_losses_3c = loss_entity_3c(dftb_calculator, data_delegate)
    check_3c = torch.allclose(total_loss_3c, loss_q_ref_3a * 0.7
                              + loss_fermi_ref_3b * 0.3)

    assert check_3c, ('Loss calculated by different loss functions with weights '
                     'for different properties does not give the correct value.')

    # Check 3d: different loss functions with weights for properties and weights for systems
    loss_entity_3d = Loss(prediction_delegate, reference_delegate, system_weight_delegate,
                          loss_functions={"q_final_atomic": l1_loss,
                                          "fermi_energy": mse_loss},
                          loss_weights={"q_final_atomic": 0.7,
                                        "fermi_energy": 0.3})
    total_loss_3d, raw_losses_3d = loss_entity_3d(dftb_calculator, data_delegate)
    loss_func_l1_ref2 = nn.L1Loss(reduction='none')
    loss_func_mse_ref2 = nn.MSELoss(reduction='none')
    weights = torch.tensor([0.2, 0.3, 0.5], device=device).unsqueeze(-1)
    loss_q_ref_3d = (loss_func_l1_ref2(dftb_calculator.q_final_atomic,
                                       data_delegate["q_final_atomic"]) * weights).mean()
    loss_fermi_ref_3d = (loss_func_mse_ref2(dftb_calculator.fermi_energy,
                                            data_delegate["fermi_energy"]) * weights).mean()
    check_3d_1 = torch.allclose(raw_losses_3d["q_final_atomic"], loss_q_ref_3d)
    check_3d_2 = torch.allclose(raw_losses_3d["fermi_energy"], loss_fermi_ref_3d)
    check_3d_3 = torch.allclose(total_loss_3d, loss_q_ref_3d * 0.7
                                + loss_fermi_ref_3d * 0.3)
    check_3d = check_3d_1 and check_3d_2 and check_3d_3

    assert check_3d, ('Loss calculated by different loss functions with weights '
                     'for different properties and weights for systems does not '
                     'give the correct value.')


def test_loss_class(device, scc_feeds):
    """Test the functions of loss class."""
    loss_class_helper(scc_feeds, data_delegate(device), prediction_delegate,
                      reference_delegate, device)
