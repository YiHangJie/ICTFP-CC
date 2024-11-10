import warnings
import logging
import numpy as np
import scipy
from argparse import Namespace
from data.Benchmark_read_attack import Benchmark_read, Benchmark_attacked_read
from data.BETA_read_attack import Beta_read, Beta_attacked_read
from utils.log import set_logger
from utils.metrics import itr
from algorithm.method import Classifier
from algorithm.ART.feature_squeezing import FeatureSqueezing
from algorithm.ART.resample import Resample
from algorithm.ART.spatial_smoothing import SpatialSmoothing
from algorithm.ART.variance_minimization import TotalVarMin


# 忽略警告
warnings.filterwarnings("ignore")


class Experiments():

    def __init__(self, 
                dataset="Benchmark" ,       # Benchmark or BETA
                attack=0,                   # the attack amplitude, [0, 0.1, 0.2, 0.3]
                ws=2,                       # window size
                subjects=list(range(1, 36)), 
                samp_rate=250, 
                freq_num=40,                # number of stimulus
                multi_freq=5,               # number of harmonics for refenrence signal
                offset_time=0.14,           # the visual delay time
                method="CCA",               # SSVEP algorithms, [CCA, FBCCA, MSI]
                C93=False,                  # if C93
                C9n=3,                      # the parameter of C93
                denoise=False,              # if denoise using ICA
                n_components=9,             # number of ICA components
                pearson_kurt_threshold=10,  # hyperparameter for ICA denoise
                baseline="None"
                ):
        self.dataset = dataset
        self.attack = attack
        self.ws = ws
        self.subjects = subjects
        self.samp_rate = samp_rate
        self.freq_num = freq_num
        self.multi_freq = multi_freq
        self.offset_time = offset_time
        self.method = method
        self.C93 = C93
        self.C9n = C9n
        self.denoise = denoise
        self.n_components = n_components
        self.pearson_kurt_threshold = pearson_kurt_threshold
        self.opt = Namespace(**vars(self))
        self.baseline = baseline
        self.baseline_methods = {"FeatureSqueezing": FeatureSqueezing, "Resample": Resample, "SpatialSmoothing": SpatialSmoothing, "TotalVarMin": TotalVarMin}
        
        if self.dataset == "Benchmark":
            self.stim_event_freq = [8, 9, 10, 11, 12, 13, 14, 15, 8.2, 9.2, 10.2, 11.2, 12.2, 13.2,
                14.2, 15.2, 8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4, 8.6, 9.6, 10.6, 11.6,
                12.6, 13.6, 14.6, 15.6, 8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8]
            if self.attack == 0:
                self.dataread = Benchmark_read
            else:
                self.dataread = Benchmark_attacked_read
        elif self.dataset == "BETA":
            self.stim_event_freq = [8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.2, 10.4, 10.6, 10.8,
                    11.0, 11.2, 11.4, 11.6, 11.8, 12.0, 12.2, 12.4, 12.6, 12.8, 13.0, 13.2, 13.4,
                    13.6, 13.8, 14.0, 14.2, 14.4, 14.6, 14.8, 15.0, 15.2, 15.4, 15.6, 15.8, 8.0, 8.2, 8.4]
            if self.attack == 0:
                self.dataread = Beta_read
            else:
                self.dataread = Beta_attacked_read


    def run(self):
        set_logger(f"{self.dataset}_{self.attack}_{self.method}")

        # read data
        data, labels = self.dataread(subjects=self.subjects, noise_amp=self.attack)
        data = data[:, :, : round(self.ws * self.samp_rate)]

        # initilize algorithm
        classifier = Classifier(self.stim_event_freq, self.opt)

        # iterate trials and predict
        preds = classifier.classify(data)
        # print(preds)

        # calculate accuracy and ITR
        accuracy = np.mean(preds == labels)
        ITR = itr(accuracy, self.freq_num, self.ws)

        logging.info('##------------------------------##')
        logging.info(f"accuracy={accuracy}, ITR={ITR}")
        info_keys = ['dataset', 'method', 'attack', 'ws', 'C93', 'C9n', 'denoise', 'n_components', 'subjects']
        info = { key: getattr(self, key) for key in info_keys }
        logging.info(info)

        return accuracy, ITR
    
    def run_baseline(self):
        assert self.baseline is not None

        set_logger(f"{self.dataset}_{self.attack}_{self.method}")

        # read data
        data, labels = self.dataread(subjects=self.subjects, noise_amp=self.attack)
        data = data[:, :, : round(self.ws * self.samp_rate)]

        baseline = self.baseline_methods[self.baseline]()
        baseline_data, _ = baseline(data, labels)

        if self.baseline == "Resample":
            self.opt.samp_rate = baseline.sr_new
        # initilize algorithm
        classifier = Classifier(self.stim_event_freq, self.opt)

        # iterate trials and predict
        preds = classifier.classify(baseline_data)
        # print(preds)

        # calculate accuracy and ITR
        accuracy = np.mean(preds == labels)
        ITR = itr(accuracy, self.freq_num, self.ws)

        logging.info('##------------------------------##')
        logging.info(f"accuracy={accuracy}, ITR={ITR}")
        info_keys = ['dataset', 'method', 'attack', 'ws', 'C93', 'C9n', 'denoise', 'n_components', 'subjects']
        info = { key: getattr(self, key) for key in info_keys }
        logging.info(info)

        return accuracy, ITR

if __name__ == "__main__":
    Experiments()


