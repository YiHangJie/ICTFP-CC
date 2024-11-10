import numpy as np
import math

def itr(accuracy, freq_num, cal_time):
    ITR = 0
    if accuracy != 1:
        ITR = (np.log2(freq_num) + accuracy * np.log2(accuracy) + (1 - accuracy) * np.log2(
            (1 - accuracy) / (freq_num - 1))) * 60 / cal_time
    else:
        ITR = math.log2(freq_num) * 60 / cal_time

    return ITR

