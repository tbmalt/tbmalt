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
                  dim: int = 0) -> Tuple[Tensor]:
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
