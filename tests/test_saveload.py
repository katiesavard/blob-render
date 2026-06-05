#import pytest
#import numpy as np
#import numpy.testing as npt

#from blobrender.tools.basics import save_list, load_list


#@pytest.mark.parametrize(
#    "array",
#    [
#        np.array([1, 2, 3]),
#        np.array([0.1, 0.2, 0.3]),
#        np.arange(10),
#    ],
#)
#def test_save_load_roundtrip(tmp_path, array):

#    save_list(array, str(tmp_path), "arr")

#    loaded = load_list(str(tmp_path), "arr")

#    npt.assert_array_equal(array, loaded)
