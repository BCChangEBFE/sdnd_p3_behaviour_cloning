import numpy as np
from scipy import misc
import os

for name_image in os.listdir("."):
    print(name_image)
    flipped_image = misc.imread(name_image)
    flipped_image = np.fliplr(flipped_image)
    misc.imsave("flip_%s"%name_image, flipped_image)
    

