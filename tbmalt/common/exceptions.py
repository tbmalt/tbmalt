"""Custom TBMaLT exceptions.
"""

class TBMaLTError(Exception):
    """The base exception class for the TBMaLT package.

    All custom exceptions should inherit from this exception. However this may
    be raised itself if deemed necessary.

    Arguments:
        msg: The message that will be printed when the exception is raised.

    """
    def __init__(self, msg: str):
        self.msg = msg


class ConvergenceError(TBMaLTError):
    """Raised whenever an failure to converge is encountered.

    This exception is raise anywhere where a failure to converge is encountered.
    For example a SCC/SCF cycle, a fermi-level search.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
