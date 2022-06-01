from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import torch
import json

TESTDIR = Path(__file__).parent
SAMPLE = TESTDIR.joinpath("sample_data.json")

@pytest.fixture(scope="session")
def sample_labels():
    """labels tensor"""
    torch.manual_seed(0)
    labels = torch.rand(200, 3)
    labels = np.argmax(labels, axis=1).flatten()
    # torch.manual_seed(0)
    # return (torch.rand(200) < 0.5).int()
    return labels


@pytest.fixture(scope="session")
def sample_data():
    """sample user/item dataset"""
    with open(SAMPLE) as f:
        df = pd.DataFrame(json.load(f))
    return df
