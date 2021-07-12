#!/usr/bin/env python
# -*- coding: utf-8 -*-

import imageio
import sys, os, glob
import numpy as np
from scipy import ndimage
# import matplotlib.pyplot as plt

fns = sorted(glob.glob('animate_output/*.png'))

with imageio.get_writer('animate.gif', mode='I', duration=1/12) as writer:
    for fn in fns:
        img = np.array(imageio.imread(fn))
        img = img[:, 420:-420]
        img = img.reshape(img.shape[0]//2, 2, img.shape[1]//2, 2, 3).mean((1, 3)).astype('u1')
        writer.append_data(img)

# plt.imshow(img)
# plt.show()
