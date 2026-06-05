import pytest
import numpy as np
import numpy.testing as npt


@pytest.mark.parametrize(
    "image",
    [
        np.add.outer(np.arange(10), np.arange(10)),
    ],
)
def test_deres_array_preserves_square_pixels(image):
    """
    If row and column resolutions are already equal,
    deres_array_check should leave the array unchanged.
    """

    from blobrender.fits_conversion import deres_array_check

    result, output = deres_array_check(
        image,
        verbose=False,
        output_string=""
    )

    npt.assert_array_equal(result, image)

    assert "square" in output.lower()
    
