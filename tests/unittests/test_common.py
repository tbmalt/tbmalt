# -*- coding: utf-8 -*-
"""Unit tests associated with `tbmalt.common.__init__`."""
import numpy as np
import pytest
import torch
from tests.test_utils import fix_seed
from tbmalt.common import split_by_size, cached_property


@fix_seed
def test_split_by_size(device):
    """Tests functionality of `split_by_size`.

    Device and gradient tests are not needed as the results is just a view of
    the input. Single and batch test are also not carried out as they are not
    applicable to this function.

    Notes:
        The `split_by_size` function will eventually get phased out once the
        pytorch `split_with_sizes` function is fully supported in the main
        pytorch branch.
    """
    for _ in range(10):
        a = torch.rand(10, 10, 10, device=device)
        indices = list(np.random.choice(range(1, 4), 2, replace=False))
        indices.append(int(10 - sum(indices)))
        dim = int(torch.randint(0, 3, (1,))[0])
        ref = torch.split_with_sizes(a, indices, dim=dim)
        prd = split_by_size(a, torch.tensor(indices, device=device), dim=dim)

        # Check that the split operation proceeded as anticipated
        check_1 = all([torch.allclose(i, j) for i, j in zip(prd, ref)])

        assert check_1, 'Tensor split operation failed'

        # Check that the returns are on the correct device
        check_2 = all([i.device == device for i in prd])

        assert check_2, f'Device persistence check failed'

    # Check 3: Ensure assert errors are raised for invalid chunk sizes
    with pytest.raises(AssertionError, match='Sum of split sizes fails*'):
        split_by_size(torch.tensor([1, 1, 1, 1]), [1])


class SomeClassWithCachedProperty:
    def __init__(self):
        self.first_dependent_tensor = torch.tensor([1.234])
        self.second_dependent_tensor = torch.tensor([4.321])
        self.some_unrelated_tensor = torch.tensor([0.])

        self.function_call_count = 0

    @cached_property('first_dependent_tensor', 'second_dependent_tensor')
    def some_cached_property(self):
        self.function_call_count += 1
        return self.first_dependent_tensor * self.second_dependent_tensor


def cached_property_helper(instance: SomeClassWithCachedProperty):

    # Store the call count value before making the call to the property
    call_count_at_start = instance.function_call_count

    # Retrieve the cached property
    property_value = instance.some_cached_property

    # Store the call count value after making the call to the property
    call_count_at_end = instance.function_call_count

    # Check that the returned cached property value is correct
    expected_property_value = (instance.first_dependent_tensor
                               * instance.second_dependent_tensor)
    property_value_is_correct = property_value == expected_property_value
    assert property_value_is_correct, "Cached property value is incorrect"

    # If any of the cached property's dependencies have been modified then the
    # cache will need to be rebuilt. In which case a single call is expected
    # to be made to the associated constructor function.
    if (call_count_at_end - call_count_at_start) == 1:
        cache_updated = True
    # If none of the dependence have changed then one would expect no calls to
    # be made to the constructor.
    elif call_count_at_end == call_count_at_start:
        cache_updated = False
    # If the difference between the call-counts is anything other than one or
    # zero then we have a problem.
    else:
        raise ValueError(
            "An unexpected number of cache reconstruction calls were made")

    return cache_updated


def test_cached_property():
    """Perform test of the `cached_property` wrapper method.

    Notes:
        Device testing is not performed here as the device on which a return
        is provided has nothing to do with the property caching operation.

    """

    test_class = SomeClassWithCachedProperty()

    # Ensure that the cached property calls out to the constructor method and
    # returns the correct result.

    function_called = cached_property_helper(test_class)
    assert function_called, "Cache did not call to the constructor method"

    # Check that cached properties are not reconstructed every time a call is
    # made to retrieve them.

    # Call out to property to ensure that the cache has been build
    _ = test_class.some_cached_property
    function_called = cached_property_helper(test_class)
    assert not function_called, "Cache is being rebuilt at every call"

    # Confirm that the cache is not rebuild when a non-dependant tensor is
    # changed.
    _ = test_class.some_cached_property
    test_class.some_unrelated_tensor = torch.tensor([0.])
    for _ in range(10):  # Cause some changes to the Tensor._version attribute
        test_class.some_unrelated_tensor[0] = 1.0
    function_called = cached_property_helper(test_class)

    assert not function_called, \
        "Cache is being rebuilt after an unrelated tensor was modified"

    # Verify that the cached property is updated when a dependent tensor is
    # modified, reassigned, or has its `requires_grad` option changed.
    _ = test_class.some_cached_property

    # Modification test
    cph = cached_property_helper

    test_class.first_dependent_tensor[0] = 1.0
    assert cph(test_class), "Cache failed to rebuild on tensor modification"

    test_class.first_dependent_tensor = torch.tensor([1.0])
    assert cph(test_class), "Cache failed to rebuild on tensor reassignment"

    test_class.first_dependent_tensor.requires_grad = True
    assert cph(test_class), "Cache failed to rebuild on requires_grad change"

    test_class.second_dependent_tensor[0] = 1.0
    assert cph(test_class), "Cache failed to rebuild on tensor modification"

    test_class.second_dependent_tensor = torch.tensor([1.0])
    assert cph(test_class), "Cache failed to rebuild on tensor reassignment"

    test_class.second_dependent_tensor.requires_grad = True
    assert cph(test_class), "Cache failed to rebuild on requires_grad change"

    function_called = cached_property_helper(test_class)

    assert not function_called, \
        "Cache is being spuriously rebuilt"
