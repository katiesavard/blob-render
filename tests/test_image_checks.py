#import pytest

#from blobrender.tools.image_checks import check_image_pixelsize


#@pytest.mark.parametrize(
#    "pixel_size,expected",
#    [
#        (0.01, True),
#        (0.1, True),
#        (100, False),
#    ]
#)
#def test_pixel_size_cases(pixel_size, expected):

#    good, _ = check_image_pixelsize(
#        pixel_size_arcsec=pixel_size,
#        max_baseline_m=1000,
#        freq_hz=1.4e9,
#    )

#    assert good == expected
