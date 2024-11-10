import numpy as np
import scipy
import scipy.special
import warnings
from itertools import combinations
from algorithm.ICA import ICA_Denoise
import multiprocessing

np.set_printoptions(suppress=True)
np.random.seed(0)

class Classifier():
    def __init__(self, stim_event_freq, opt):
        # 正余弦参考信号
        self.stim_event_freq = stim_event_freq
        self.fb_coefs = [a ** (-1.25) + 0.25 for a in range(1, 20+1)]
        # print(self.fb_coefs)
        self.nbands = 5
        self.multi_freq = opt.multi_freq
        self.samp_rate = opt.samp_rate
        self.freq_num = opt.freq_num
        self.ws = opt.ws
        self.offset = opt.offset_time
        self.offset_len = round(self.offset * self.samp_rate)
        self.T = int(self.samp_rate * self.ws) - int(self.samp_rate * self.offset)
        self.target_template_set = self.get_Reference_Signal(self.multi_freq, self.stim_event_freq)
        self.method = opt.method
        self.if_C93 = opt.C93
        self.C9n = opt.C9n
        self.if_denoise = opt.denoise
        self.n_components = opt.n_components
        self.pearson_kurt_threshold = opt.pearson_kurt_threshold
        self.denoiser = ICA_Denoise(self.ws, self.samp_rate)

    def get_Reference_Signal(self, num_harmonics, targets):
        reference_signals = []
        t = np.arange(0, (self.T / self.samp_rate), step=1.0 / self.samp_rate)
        for f in targets:
            reference_f = []
            for h in range(1, num_harmonics + 1):
                reference_f.append(np.sin(2 * np.pi * h * f * t)[0:self.T])
                reference_f.append(np.cos(2 * np.pi * h * f * t)[0:self.T])
            reference_signals.append(reference_f)
        reference_signals = np.asarray(reference_signals)
        # print("reference's shape: ", np.shape(reference_signals))
        return reference_signals
    
    # 预处理
    def pre_filter(self, data):
        # 滤波
        f0 = 50
        q = 35
        b, a = scipy.signal.iircomb(f0, q, ftype='notch', fs=self.samp_rate)
        return scipy.signal.filtfilt(b, a, data)

    def filterbank(self, eeg, fs, idx_fb):
        if idx_fb == None:
            warnings.warn('stats:filterbank:MissingInput ' \
                        + 'Missing filter index. Default value (idx_fb = 0) will be used.')
            idx_fb = 0
        elif (idx_fb < 0 or 9 < idx_fb):
            raise ValueError('stats:filterbank:InvalidInput ' \
                            + 'The number of sub-bands must be 0 <= idx_fb <= 9.')

        if (len(eeg.shape) == 2):
            num_chans = eeg.shape[0]
            num_trials = 1
        else:
            num_chans, _, num_trials = eeg.shape

        # Nyquist Frequency = Fs/2N
        Nq = fs / 2

        passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
        stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
        Wp = [passband[idx_fb] / Nq, 90 / Nq]
        Ws = [stopband[idx_fb] / Nq, 100 / Nq]
        [N, Wn] = scipy.signal.cheb1ord(Wp, Ws, 3, 40)  # band pass filter StopBand=[Ws(1)~Ws(2)] PassBand=[Wp(1)~Wp(2)]
        # print(N, Wn)
        [B, A] = scipy.signal.cheby1(N, 0.5, Wn, 'bandpass')  # Wn passband edge frequency

        y = np.zeros(eeg.shape)
        if (num_trials == 1):
            for ch_i in range(num_chans):
                # apply filter, zero phass filtering by applying a linear filter twice, once forward and once backwards.
                # to match matlab result we need to change padding length
                y[ch_i, :] = scipy.signal.filtfilt(B, A, eeg[ch_i, :], padtype='odd', padlen=3 * (max(len(B), len(A)) - 1))

        else:
            for trial_i in range(num_trials):
                for ch_i in range(num_chans):
                    y[ch_i, :, trial_i] = scipy.signal.filtfilt(B, A, eeg[ch_i, :, trial_i], padtype='odd',
                                                                padlen=3 * (max(len(B), len(A)) - 1))
        return y

    def find_Synchronization_Index(self, X, Y):
        num_freq = Y.shape[0]
        num_harm = Y.shape[1]
        result = np.zeros(num_freq)
        for freq_idx in range(0, num_freq):
            y = Y[freq_idx]
            X = X[:] - np.mean(X).repeat(self.T * len(X)).reshape(len(X), self.T)
            X = X[:] / np.std(X).repeat(self.T * len(X)).reshape(len(X), self.T)

            y = y[:] - np.mean(y).repeat(self.T * num_harm).reshape(num_harm, self.T)
            y = y[:] / np.std(y).repeat(self.T * num_harm).reshape(num_harm, self.T)

            c11 = (1 / self.T) * (np.dot(X, X.T))
            c22 = (1 / self.T) * (np.dot(y, y.T))
            c12 = (1 / self.T) * (np.dot(X, y.T))
            c21 = c12.T

            C_up = np.column_stack([c11, c12])
            C_down = np.column_stack([c21, c22])
            C = np.row_stack([C_up, C_down])

            # print("c11:", c11)
            # print("c22:", c22)

            v1, Q1 = np.linalg.eig(c11)
            v2, Q2 = np.linalg.eig(c22)
            v1 = np.abs(v1)
            v2 = np.abs(v2)
            V1 = np.diag(v1 ** (-0.5))
            V2 = np.diag(v2 ** (-0.5))

            C11 = np.dot(np.dot(Q1, V1.T), np.linalg.inv(Q1))
            C22 = np.dot(np.dot(Q2, V2.T), np.linalg.inv(Q2))

            # print("Q1 * Q1^(-1):", np.dot(Q1, np.linalg.inv(Q1)))
            # print("Q2 * Q2^(-1):", np.dot(Q2, np.linalg.inv(Q2)))

            U_up = np.column_stack([C11, np.zeros((len(X), num_harm))])
            U_down = np.column_stack([np.zeros((y.shape[0], len(X))), C22])
            U = np.row_stack([U_up, U_down])
            R = np.dot(np.dot(U, C), U.T)

            R = np.nan_to_num(R, nan=0, posinf=0, neginf=0)
            eig_val, _ = np.linalg.eig(R)
            # print("eig_val:", eig_val, eig_val.shape)
            E = eig_val / np.sum(eig_val)
            S = 1 + np.sum(E * np.log(E)) / np.log(len(X) + num_harm)
            result[freq_idx] = S

        return result

    def CCA(self, data, target_template_set):
        p = []
        data = data.T
        # qr分解,data:length*channel
        [Q_temp, R_temp] = np.linalg.qr(data)
        for template in target_template_set:
            template = template[:, 0:data.shape[0]]
            template = template.T
            [Q_cs, R_cs] = np.linalg.qr(template)
            data_svd = np.dot(Q_temp.T, Q_cs)
            [U, S, V] = np.linalg.svd(data_svd)
            rho = np.dot(self.fb_coefs[: S.shape[0]], S.T)
            p.append(rho)
        result = p.index(max(p))+1
        return result

    def FBCCA(self, dataCCA, target_template_set):
        _, num_smpls = dataCCA.shape  # 40 taget (means 40 fre-phase combination that we want to predict)
        y_ref = target_template_set
        # result matrix
        r = np.zeros((self.nbands, len(target_template_set)))
        # deal with one target a time
        for fb_i in range(self.nbands):  # filter bank number, deal with different filter bank
            testdata = self.filterbank(dataCCA, self.samp_rate, fb_i)  # data after filtering
            testdata = testdata.T
            [Q_temp, R_temp] = np.linalg.qr(testdata)
            for class_i in range(len(target_template_set)):
                template = np.squeeze(y_ref[class_i])
                template = template[:, 0:testdata.shape[0]]
                template = template.T
                [Q_cs, R_cs] = np.linalg.qr(template)
                data_svd = np.dot(Q_temp.T, Q_cs)
                [U, S, V] = np.linalg.svd(data_svd)
                rho = np.dot(self.fb_coefs[: S.shape[0]], S.T)
                r[fb_i, class_i] = rho
        p_FBCCA = np.dot(self.fb_coefs[: self.nbands], r).tolist()
        result = p_FBCCA.index(max(p_FBCCA)) + 1
        return result

    def MSI(self, test_data, reference_signals):
        
        result = self.find_Synchronization_Index(test_data, reference_signals)
        predicted_class = np.argmax(result) + 1

        return predicted_class

    def classify(self, test_data):
        preds = []
        for trial in range(len(test_data)):
            print(f"\rclassifying No.{trial+1} sample ......", end="")
            
            use_data = test_data[trial]
            trial_result = 0

            if self.if_denoise:
                use_data, _ = self.denoiser.ICA_denoise(use_data, self.pearson_kurt_threshold, n_components=self.n_components)

            # preprocess
            use_data = self.pre_filter(use_data)

            if not self.if_C93:
                if self.method == "MSI":
                    trial_result = self.MSI(use_data[:, self.offset_len:], self.target_template_set)
                elif self.method == "FBCCA":
                    trial_result = self.FBCCA(use_data[:, self.offset_len:], self.target_template_set)
                elif self.method == "CCA":
                    trial_result = self.CCA(use_data[:, self.offset_len:], self.target_template_set)
            else:
                trial_result = self.C93(use_data)

            preds.append(trial_result)
        print("\r", end="")
        return preds
    
    def C93(self, trial_data):
        channel_set = list(combinations(list(range(trial_data.shape[0])), self.C9n))
        votes = []
        for channels in channel_set:
            channels = list(channels)
            use_data = trial_data[channels]

            if self.method == "MSI":
                trial_result = self.MSI(use_data[:, self.offset_len:], self.target_template_set)
            elif self.method == "FBCCA":
                trial_result = self.FBCCA(use_data[:, self.offset_len:], self.target_template_set)
            elif self.method == "CCA":
                trial_result = self.CCA(use_data[:, self.offset_len:], self.target_template_set)

            votes.append(trial_result)
        
        pred = max(votes, key=votes.count)
        return pred
    
    def cal_com_corr(self, test_data):
        noise_corrs = []
        common_corrs = []

        for trial in range(len(test_data)):
            print(f"\ranalysing No.{trial+1} sample ......", end="")
            
            use_data = test_data[trial]
            success, max_corrs, other_corrs = self.denoiser.ICA_noise_analysis(use_data)
            if success:
                noise_corrs += max_corrs
                common_corrs += other_corrs

        print("\r", end="")
        return noise_corrs, common_corrs
