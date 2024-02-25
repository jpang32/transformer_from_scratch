import numpy as np

from src.softmax import Softmax
import pytest
from unittest.mock import patch, MagicMock


FILEPATH = "src.softmax"

@pytest.mark.parametrize(
    "layer_output, expected_output",
    [([0, 0.9, 0.1], [0.219, 0.539, 0.242]),
     ([0.25, 1.23, -0.8], [0.249, 0.664, 0.087]),
     ([0.25,1.23,-0.8], [ 0.368,  1.809, -1.176])
     ]
)
@patch(f"{FILEPATH}.Layer")
def test_softmax_forward(mock_layer, layer_output, expected_output):
    mock_layer.forward.return_value = np.array(layer_output)
    mock_input = MagicMock()
    mock_input.ndim = 1

    softmax = Softmax(layer=mock_layer)
    output = softmax.forward(inputs=mock_input)

    # Test that output of softmax is close to these values
    assert np.allclose(np.array(expected_output), output, atol=1e04)
