import pytest
import numpy as np
import numpy.testing as npt


@pytest.mark.parametrize(
    "test, expected",
    [
        (),
        ()
    ])
def test_deres_array(test, expected):
    """
    Test the deres_array function with different inputs.
    """
    from blobrender.fits_conversion import deres_array_check
    # Call the function with the test input
    image, output = deres_array_check(test)

    # Check if the result matches the expected output
    npt.assert_array_equal(image, expected)
    # Check if the result matches the expected output
    
