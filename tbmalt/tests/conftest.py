import pytest
import torch

# Default must be set to float64 otherwise gradcheck will not function
torch.set_default_dtype(torch.float64)

def pytest_addoption(parser):
    """Set up command line options."""
    parser.addoption(  # Enable device selection
        "--device", action="store", default="cpu",
        help="specify test device (cpu/cuda/etc/...)"
    )

    parser.addoption(  # Should more comprehensive gradient test be performed?
        "--detect-anomaly", action='store_true',
        help='this flag enables more comprehensive, but time consuming, '
             'gradient tests.'
    )


@pytest.fixture
def device(request) -> torch.device:
    """Defines the device on which each test should be run.

    Returns:
        device: The device on which the test will be run.

    """
    # Device checks require CPU to be specified *without* a device number and
    # and cuda to be specified *with* one.
    device_name = request.config.getoption("--device")
    if device_name == 'cuda':
        return torch.device('cuda:0')
    else:
        return torch.device(device_name)


def pytest_configure(config):
    """Pytest configuration hook."""
    # Check if the "--detect-anomaly" flag was passed, if so then turn on
    # autograd anomaly detection.
    if config.getoption('--detect-anomaly'):
        torch.autograd.set_detect_anomaly(True)


def __sft(self):
    """Alias for calling .cpu().numpy()"""
    return self.cpu().numpy()


torch.Tensor.sft = __sft
