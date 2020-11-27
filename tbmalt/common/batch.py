"""Helper functions for batch operations.

This module contains classes and helper functions associated with batch
construction, handling and maintenance.
"""
from typing import Optional, Any, Tuple, List, Union
import numpy as np
import torch
Tensor = torch.Tensor


def pack(tensors: List[Tensor],
         axis: int = 0, value: Any = 0,
         size: Optional[Union[Tuple[int], torch.Size]] = None
         ) -> Tensor:
    """Pad and pack a sequence of tensors together.

    Pad a list of variable length tensors with zeros, or some other value, and
    pack them into a single tensor.

    Arguments:
        tensors: List of tensors to be packed, all with identical dtypes.
        axis: Axis along which tensors should be packed; 0 for first axis -1
            for the last axis, etc. This will be a new dimension. [DEFAULT=0]
        value: The value with which the tensor is to be padded. [DEFAULT=0]
        size: Size of each dimension to which tensors should be padded. This
            This to the largest size encountered along each dimension.

    Returns:
        packed_tensors: Input tensors padded and packed into a single tensor.

    Notes:
        ``packed_tensors`` maintains the same order as ``tensors``. This
        is faster & more flexible than the internal pytorch pack & pad
        functions (at this particularly task).

    Examples:

        >>> from tbmalt.common.batch import pack
        >>> import torch
        >>> a, b, c = torch.rand(2,2), torch.rand(3,3), torch.rand(4,4)
        >>> abc_packed_a = pack([a, b, c])
        >>> print(abc_packed_a.shape)
        torch.Size([3, 4, 4])
        >>> abc_packed_b = pack([a, b, c], axis=1)
        >>> print(abc_packed_b.shape)
        torch.Size([4, 3, 4])
        >>> abc_packed_c = pack([a, b, c], axis=-1)
        >>> print(abc_packed_c.shape)
        torch.Size([4, 4, 3])

    """
    # Gather some general setup info
    count, device, dtype = len(tensors), tensors[0].device, tensors[0].dtype

    # Identify the maximum size, if one was not specified.
    size = np.max([i.shape for i in tensors], 0) if size is None else size

    # Tensor to pack into, filled with padding value.
    padded = torch.full((count, *size), value, dtype=dtype, device=device)

    # Loop over & pack "tensors" into "padded". A proxy index "n" must be used
    # for assignments rather than a slice to prevent in-place errors.
    for n, source in enumerate(tensors):
        # Slice operations not elegant but they are dimension agnostic & fast.
        padded[(n, *[slice(0, s) for s in source.shape])] = source

    # If "axis" was anything other than 0, then "padded" must be permuted.
    if axis != 0:
        # Resolve relative any axes to their absolute equivalents to maintain
        # expected slicing behaviour when using the insert function.
        axis = padded.dim() + 1 + axis if axis < 0 else axis

        # Build a list of axes indices; but omit the axis on which the data
        # was concatenated (i.e. 0).
        ax = list(range(1, padded.dim()))

        # Re-insert the concatenation axis as specified
        ax.insert(axis, 0)

        # Perform the permeation
        padded = padded.permute(ax)

    return padded
