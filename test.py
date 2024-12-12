import simulate_incoherent
import utils
import numpy as np

params = {
    'length': 1.3e-3,
    'lambda': 555e-9,
    'wxp': 1e-3,
    'zxp': 123.8e-3,
    'm': 10,
}

img = np.zeros((256, 256))
ctf, otf, psf = simulate_incoherent.incoh_otf(img, params)
utils.single_show(ctf, "ctf")
utils.single_show(otf, "otf")
utils.single_show(psf, "psf")
