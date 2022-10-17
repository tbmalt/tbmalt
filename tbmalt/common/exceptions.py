# -*- coding: utf-8 -*-
"""Custom TBMaLT exceptions."""
import torch
from torch import Tensor


class TbmaltError(Exception):
    """The base exception class for the TBMaLT package.

    All custom exceptions should inherit from this exception. However this may
    be raised itself if deemed necessary.

    Arguments:
        msg: The message that will be printed when the exception is raised.

    """
    def __init__(self, msg: str):
        self.msg = msg


class ConvergenceError(TbmaltError):
    """Raised whenever an failure to converge is encountered.

    This is raise wherever a failure to converge is encountered. Most commonly
    during the SCC/SCF cycles or the fermi-level search.

    Arguments:
        msg: message displayed when throwing the exception.
        failure_mask: A tensor indicating which systems failed to converge
            (True) & which converged successfully (False).

    Notes:
        The number of systems which failed to converge will be appended to the
        end of the message automatically. Furthermore, the exact systems which
        failed to converge can be identified form the failure mask attribute.
    """

    def __init__(self, msg: str, failure_mask: Tensor):
        self.failure_mask = failure_mask
        super().__init__(msg)

    def __str__(self):
        msk = torch.atleast_1d(self.failure_mask)
        return self.msg + f' ({msk.count_nonzero()}/{(len(msk))} failed)'
