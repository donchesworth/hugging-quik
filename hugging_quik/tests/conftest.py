import pytest
import numpy as np
import torch


@pytest.fixture(scope="session")
def sample_labels():
    """labels tensor"""
    torch.manual_seed(0)
    labels = torch.rand(200, 3)
    labels = np.argmax(labels, axis=1).flatten()
    # torch.manual_seed(0)
    # return (torch.rand(200) < 0.5).int()
    return labels
