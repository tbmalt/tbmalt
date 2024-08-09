# -*- coding: utf-8 -*-
"""A collection of useful code abstractions.

All modules that are not specifically associated with any one component of the
code, such as generally mathematical operations, are located here.
"""
from typing import Tuple, Union, List, Callable
from functools import wraps
import torch
from torch import Tensor

# Types
float_like = Union[Tensor, float]
bool_like = Union[Tensor, bool]


def cached_property(*dependency_names: str):
    """Decorator to handle caching of properties based on their dependencies.

    Before returning a cached property, the decorator checks whether any
    dependencies have been modified or reassigned. If so, the cache is rebuilt.
    Note that this currently only supports torch tensors.


    Arguments:
        dependency_names: Names of the class attributes that are considered to
            be dependencies. These attributes must be torch tensors.
    """

    def decorator(method):
        """Wrap the method to check dependencies before returning its value."""

        @property
        @wraps(method)
        def wrapper(self):

            rebuild_cache = False

            # Look over the dependency names.
            for dependency_name in dependency_names:

                # Get the dependency (i.e. self.some_dependency_tensor)
                dependency = getattr(self, dependency_name)

                # Ensure the dependency is a torch tensor.
                if not isinstance(dependency, Tensor):
                    raise TypeError(
                        'Only torch tensors can be cache dependencies')

                # Identify the current version of the dependency tensor and
                # what version was used to last construct the cache. A mismatch
                # in version numbers indicates the contents of the dependency
                # changed since the cache was last build.
                current_version = dependency._version
                cache_version = getattr(
                    wrapper.fget, f'cache_version_{dependency_name}')
                version_mismatch = current_version != cache_version

                # Repeat this process for the dependency tensor object's ID.
                # This is used to determine if tensor itself changed since the
                # cache was last constructed.
                current_id = id(dependency)
                cache_id = getattr(
                    wrapper.fget, f'cache_id_{dependency_name}')
                id_mismatch = current_id != cache_id

                # Repeat the process for the dependency tensors requires grad
                # attribute. This is important to ensure that properties are
                # reconstructed allowing for the graph to be rebuilt if users
                # change whether or not they are tracking the gradient through
                # a specific property.
                current_requires_grad = dependency.requires_grad
                cache_requires_grad = getattr(
                    wrapper.fget, f'cache_requires_grad_{dependency_name}')
                requires_grad_mismatch = (
                        current_requires_grad != cache_requires_grad)

                # Has the dependency tensor or its contents changed since the
                # cache was last constructed?
                if version_mismatch or id_mismatch or requires_grad_mismatch:
                    # Earmark the cache for reconstruction.
                    rebuild_cache = True

                    # Update the cache version numbers. It is fine to to this
                    # here as the cache will get updated immediately after this
                    # loop has finished.
                    setattr(wrapper.fget, f'cache_version_{dependency_name}',
                            current_version)

                    setattr(wrapper.fget, f'cache_id_{dependency_name}',
                            current_id)

                    setattr(
                        wrapper.fget, f'cache_requires_grad_{dependency_name}',
                        current_requires_grad)

                    # This loop is not broken out of here as one must ensure
                    # that version numbers for other dependencies are updated.

            # If a version mismatch has been detected in any of the dependencies
            # then the cache must be recomputed.
            if rebuild_cache:
                setattr(wrapper.fget, 'cache', method(self))

            # Return the contents of the cache
            return getattr(wrapper.fget, 'cache')

        # Add attributes to the *property* for tracking i) the version numbers,
        # ii) the id's of its dependencies, and iii) the requires gradient
        # status of the tensor.
        for dependency_name_o in dependency_names:
            setattr(wrapper.fget, f'cache_version_{dependency_name_o}', -1)
            setattr(wrapper.fget, f'cache_id_{dependency_name_o}', -1)
            setattr(wrapper.fget, f'cache_requires_grad_{dependency_name_o}',
                    False)

        # Add an attribute for storing the cached value of the property.
        setattr(wrapper.fget, f'cache', None)

        return wrapper

    return decorator


def split_by_size(tensor: Tensor, sizes: Union[Tensor, List[int]],
                  dim: int = 0) -> Tuple[Tensor, ...]:
    """Splits a tensor into chunks of specified length.

    This function takes a tensor & splits it into `n` chunks, where `n` is the
    number of entries in ``sizes``. The length of the `i'th` chunk is defined
    by the `i'th` element of ``sizes``.

    Arguments:
        tensor: Tensor to be split.
        sizes: Size of each chunk.
        dim: Dimension along which to split ``tensor``.

    Returns:
        chunked: Tuple of tensors viewing the original ``tensor``.

    Examples:
        Tensors can be sequentially split into multiple sub-tensors like so:

        >>> from tbmalt.common import split_by_size
        >>> a = torch.arange(10)
        >>> print(split_by_size(a, [2, 2, 2, 2, 2]))
        (tensor([0, 1]), tensor([2, 3]), tensor([4, 5]), tensor([6, 7]), tensor([8, 9]))
        >>> print(split_by_size(a, [5, 5]))
        tensor([0, 1, 2, 3, 4]), tensor([5, 6, 7, 8, 9]))
        >>> print(split_by_size(a, [1, 2, 3, 4]))
        (tensor([0]), tensor([1, 2]), tensor([3, 4, 5]), tensor([6, 7, 8, 9]))

    Notes:
        The resulting tensors ``chunked`` are views of the original tensor and
        not copies. This was created as no analog existed natively within the
        pytorch framework. However, this will eventually be removed once the
        pytorch function `split_with_sizes` becomes operational.

    Raises:
        AssertionError: If number of elements requested via ``split_sizes``
            exceeds the number of elements present in ``tensor``.
    """
    # Looks like it returns a tuple rather than a list
    if dim < 0:  # Shift dim to be compatible with torch.narrow
        dim += tensor.dim()

    # Ensure the tensor is large enough to satisfy the chunk declaration.
    size_match = tensor.shape[dim] == sum(sizes)
    assert size_match, 'Sum of split sizes fails to match tensor length ' \
                       'along specified dim'

    # Identify the slice positions
    splits = torch.cumsum(torch.tensor([0, *sizes]), dim=0)[:-1]

    # Return the sliced tensor. use torch.narrow to avoid data duplication
    return tuple(torch.narrow(tensor, dim, start, length)
                 for start, length in zip(splits, sizes))


def unique(x: Tensor, return_index=False, sorted=True, return_inverse=False,
           return_counts=False, dim=None):
    """Find the unique elements of a tensor.

    Note that the code for this method is taken directly from a post provided
    by user @wjaekim on PyTorch GitHub issue #36748, and the documentation is
    taken directly from the numpy documentation. This function is intended to
    act only a temporary fix until PyTorch supports the ``return_index`` key-
    word argument in their internal method.

    Returns the sorted unique elements of an array. There are three optional
    outputs in addition to the unique elements:

        - the indices of the input array that give the unique values.
        - the indices of the unique array that reconstruct the input tensor.
        - the number of times each unique value comes up in the input tensor.

    Arguments:
        x: Input array. Unless dims is specified, this will be flattened if it
            is not already 1-D.
        return_index: If True, also return the indices of ar (along the specified
            dimension, if provided, or in the flattened tensor) that result in the
            unique tensor.
        return_inverse: If True, also return the indices of the unique tensor (for
            the specified dimension, if provided) that can be used to reconstruct
            the supplied tensor.
        return_counts: If True, also return the number of times each unique item
            appears in the input tensor.
        dims: The dimension to operate on. If None, the tensor will be flattened.
            If an integer, the sub-tensors indexed by the given dimension will be
            flattened and treated as the elements of a 1-D tensor with the
            dimension of the given dimension. The default is None.

    Returns:
        unique: The sorted unique values.
        unique_indices: The indices of the first occurrences of the unique
            values in the original tensor. Only provided if return_index is True.
        unique_inverse: The indices to reconstruct the original array from
            the unique tensor. Only provided if return_inverse is True.
        unique_counts: The number of times each of the unique values comes up
            in the original tensor. Only provided if return_counts is True.

    """
    if return_index or (not sorted and dim is not None):

        unique_v, inverse, counts = torch.unique(
            x, sorted=True, return_inverse=True, return_counts=True, dim=dim)

        inv_sorted, inv_argsort = inverse.flatten().sort(stable=True)

        tot_counts = torch.cat(
            (counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]

        index = inv_argsort[tot_counts]

        if not sorted:
            index, idx_argsort = index.sort()
            unique_v = (unique_v[idx_argsort] if dim is None else
                        torch.index_select(unique_v, dim, idx_argsort))
            if return_inverse:
                idx_tmp = idx_argsort.argsort()
                inverse.flatten().index_put_(
                    (inv_argsort,), idx_tmp[inv_sorted])
            if return_counts:
                counts = counts[idx_argsort]

        ret = (unique_v,)
        if return_index:
            ret += (index,)
        if return_inverse:
            ret += (inverse,)
        if return_counts:
            ret += (counts,)
        return ret if len(ret) > 1 else ret[0]

    else:
        return torch.unique(
            x, sorted=sorted, return_inverse=return_inverse,
            return_counts=return_counts, dim=dim)