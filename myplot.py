import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt

import imageio
import os
def generate_gif(path):
    files = os.listdir(path)
    imgs = []
    paths = [os.path.join(path,file) for file in files if file[-3:] == "png"]
    paths = sorted(paths, key=lambda x: int(x.split("/")[-1][:-4]))
    for path in paths:
        imgs.append(plt.imread(path))
    kargs = { 'duration': 0.3 }
    imageio.mimsave("1.gif", imgs, "GIF", **kargs)
if __name__=="__main__":
    data = pd.read_csv("/home/xh/pj/CSE257/temp/sgm/logs/thinned_fourrooms_sgm_2021-05-31_02-44-29/evaluation.csv")
    plt.plot(data["Cleanup Steps"].to_numpy()[1:], data["Success Rate"].to_numpy()[1:])
    plt.show()
    path_to_images="./"
    generate_gif(path_to_images)