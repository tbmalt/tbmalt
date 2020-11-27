# content of conftest.py
import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--device", action="store", default="cpu", help="specify test device (cpu/cuda)"
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
    if device_name == 'cpu':
        return torch.device('cpu')
    if device_name == 'cuda':
        return torch.device('cuda:0')

