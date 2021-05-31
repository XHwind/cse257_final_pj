import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
if __name__=="__main__":
    data = pd.read_csv("/home/xh/pj/CSE257/temp/sgm/logs/thinned_fourrooms_sgm_2021-05-31_02-44-29/evaluation.csv")
    plt.plot(data["Cleanup Steps"].to_numpy()[1:], data["Success Rate"].to_numpy()[1:])
    plt.show()
    pdb.set_trace()