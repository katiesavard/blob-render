import numpy as np
import dart as dt
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """
    Perform a quick render against a standard mesh, return pixel RMS
    """
    print("starting regression test...")
    print("tracing 500^2 image on 500^3 mesh")
    l = 0.5
    res = 500
    x = np.linspace(-l, l, res)
    y = np.linspace(-l, l, res)
    z = np.linspace(-l, l, res)
    xx, yy, zz = np.meshgrid(x, y, z)
    r2 = xx ** 2 + yy ** 2 + 0.25 * zz ** 2
    trace = 1 / (r2 + 1e-3)

    dims = np.shape(trace)
    bbox = [[-l,l], [-l,l], [-l,l]]
    mb = dt.MeshBlock(bbox, trace)

    mesh = dt.Mesh()
    mesh.add_mb(mb)

    phi_deg = 0.005
    theta_deg = 90.005
    phi_rad = phi_deg * np.pi / 180.0
    theta_rad = theta_deg * np.pi / 180.0
    pdim = [500,500]
    sdim = [l] * 2
    screen = dt.Screen(R=l*3, theta=theta_rad, phi=phi_rad, sdim=sdim, pdim=pdim, tilt=0)
    img = screen.render(mesh, use_bake=False, verbose=True)

    standard_img = np.load("test_image.npy")
    delta = np.abs(standard_img - img)
    rms = np.sqrt(np.sum(delta ** 2) / np.size(delta))

    print("RMS pixel error with standard image  = {0}".format(rms))
    print("regression test complete.")
