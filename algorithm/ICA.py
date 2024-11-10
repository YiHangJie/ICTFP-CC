import mne
import numpy as np
import math
import h5py
from sklearn.preprocessing import minmax_scale
from mne.preprocessing import ICA
from scipy.stats import kurtosis
from scipy.stats import pearsonr, spearmanr
from scipy import spatial
from scipy import signal

if_show = False
if_FD = True


class ICA_Denoise():

    def __init__(self, ws, samp_rate):
        self.ws = ws
        self.samp_rate = samp_rate
        # 噪声频率
        self.f_list = [round(8 + 0.1 * i, 1) for i in range(80)]
        self.attack_types = {0: "sin_wave", 1: "square_wave", 2: "rec_wave", 3: "sawtooth_wave"}
        self.noise_wave_templetes = self.get_noise_wave()


    def delay_signal(self, signal, time, fs=250):
        pad_width = math.ceil(time*fs/2)

        signal = np.pad(signal, pad_width, 'constant', constant_values=0)
        signal = np.roll(signal, int(time*fs))
        signal = signal[pad_width:(len(signal)-pad_width)]
        return signal


    # 扰动相位：扰动存在0-40ms的随机相位延迟
    def wave_produce(self, mode, fs, f, time_delay, dc=0.5, amplitude=1, length=2*250, unipolar=False):
        if mode == 0:
            return self.sin_produce(amplitude, f, time_delay, length)
        elif mode == 1:
            return self.square_wave_produce(fs=fs, f=f, time_delay=time_delay, dc=dc, amplitude=amplitude, length=length, unipolar=unipolar)
        elif mode == 2:
            return self.rec_wave_produce(fs, f, time_delay=time_delay, dc=0.2, amplitude=amplitude, length=length, unipolar=unipolar)
        elif mode == 3:
            return self.sawtooth_wave_produce(fs, f, time_delay=time_delay, amplitude=amplitude, length=length)
        else:
            return None


    def sin_produce(self, amplitude = 1, f = 8.3, time_delay = 0, length = 2*250):   # 振幅, 频率, 初始相位, 数据长度
        # 生成时间序列
        t = np.linspace(0, (length-1)/250, int(length))

        # 生成正弦信号
        sin_signal = np.sin(2*np.pi*f * t) * amplitude

        sin_signal = self.delay_signal(sin_signal, time_delay)

        return sin_signal


    def square_wave_produce(self, 
        fs = 250,           # 采样率
        f = 8.3,            # 信号频率
        time_delay = 0,     # 随机延迟
        dc = 0.5,           # 占空比
        amplitude = 1,      # 幅值
        length = 2*250,     # 方波长度
        unipolar = False):   # 是否是单极方波

        # 生成时间轴
        t = np.linspace(0, (length-1.0)/fs, length)

        # 生成方波
        if unipolar:
            square_wave = np.array(np.where(np.mod(t*f, 1) < dc, 1, 0)) * amplitude
        else:
            square_wave = np.array(np.where(np.mod(t*f, 1) < dc, 1, -1)) * amplitude

        # 延时
        square_signal = self.delay_signal(square_wave, time_delay)
        return square_signal

    def rec_wave_produce(self, 
        fs = 250,           # 采样率
        f = 8.3,            # 信号频率
        time_delay = 0,     # 随机延迟
        dc = 0.2,           # 占空比
        amplitude = 1,      # 幅值
        length = 2*250,     # 方波长度
        unipolar = False):   # 是否是单极方波

        # 生成时间轴
        t = np.linspace(0, (length-1)/fs, length)

        # 生成方波
        if unipolar:
            square_wave = np.where(np.mod(t*f,1) < dc, 1, 0) * amplitude
        else:
            square_wave = np.where(np.mod(t*f,1) < dc, 1, -1) * amplitude

        # 随机延时
        square_wave = self.delay_signal(square_wave, time_delay)
        return square_wave


    def sawtooth_wave_produce(self, 
        fs = 250,           # 采样率
        f = 8.3,            # 频率
        time_delay = 0,     # 随机延迟
        amplitude = 1,      # 幅值
        length = 2*250):    # 周期数

        # 生成时间轴
        t = np.linspace(0, (length-1)/fs, length, endpoint=False)

        # 生成锯齿波
        sawtooth_wave = 2 * (f*t - np.floor(f*t)-0.5) * amplitude

        # 随机延时
        sawtooth_wave = self.delay_signal(sawtooth_wave, time_delay)
        return sawtooth_wave


    def get_noise_wave(self):
        # 生成噪声参考信号
        noise_wave_templetes = []
        for m, pfreq in enumerate(self.f_list):
            # 正弦波, 方波, 占空比为20%的矩形波, 锯齿波
            # print(f"第{j}组")
            # 4种信号*4种延迟
            wave_temp = []
            for k in range(4):
                # print(f"{attack_types[k]}")
                for time_delay in range(0, 10, 3):
                    # my_indx = int(j * 600 + m * 20 + k * 5 + time_delay / 2)
                    wave_temp.append(
                        self.wave_produce(mode=k, fs=250, f=pfreq, time_delay=time_delay * 0.004,
                                        length=int(self.ws * self.samp_rate)))
            noise_wave_templetes.append(np.array(wave_temp))
        return noise_wave_templetes


    # 衡量两个频率是否可以视为相同
    def whether_same(self, num1, num2):
        if abs(num1 - num2) <= 0.25:
            return True
        return False


    def plot_ICAfreq(self, data, fs):
        kurt = []
        max_frequency = []
        extre_freq = []
        # amp_list = []
        for i in range(len(data)):
            signal = data[i]
            n = len(signal)
            yf = np.fft.fft(signal)[:n // 2]
            xf = np.fft.fftfreq(len(signal), 1 / fs)[:n // 2]
            # print(len(xf), xf)
            # 频率为8~16Hz, index1、index2分别是
            index1 = np.where(xf < 8)[0][-1]
            index2 = np.where(xf > 16)[0][0]
            # print(xf[index1], xf[index2])
            # 横轴为46个频率点
            xf = xf[index1: index2]
            amp = (np.abs(yf) * 2 / n)[index1: index2]
            # amp_list.append(amp)

            # 拿到峰度
            this_k = kurtosis(amp) + 3
            kurt.append(this_k)
            # print("当前峰度为：", this_k)

            # 找到能量高于 均值+3*标准差 的频率
            fft_freqs = list(xf[np.where(amp > np.mean(amp) + 3 * np.std(amp))])
            # print("fft_freqs", fft_freqs)
            max_frequency.append(fft_freqs)
            extre_freq.append(xf[np.argmax(amp)])
            # 可视化
            # plt.plot(xf, amp)
            # for x in fft_freqs:
            #     plt.axvline(x=x, color='red', linestyle='--')
            # plt.xlabel(f'Frequency (B:Hz) of component{i}')
            # plt.ylabel(f'Amplitude (A) of component{i}')
            # plt.grid()
            # plt.show()

        ''' 峰度最高成分的最大频率 '''
        # indx = kurt.index(sorted(kurt, reverse=True)[0])
        # amp_list[indx] = list(amp_list[indx])
        # biggest = xf[amp_list[indx].index(max(amp_list[indx]))]
        ''' 峰度最高成分的最大频率 '''

        return extre_freq, max_frequency, sorted(kurt, reverse=True)[0], kurt


    def ICA_denoise(self, data, pearson_threshold, n_components=9):
        info = mne.create_info(
            ch_names=list(str(i) for i in range(data.shape[0])),
            ch_types="eeg",  # channel type
            sfreq=250
        )
        # use_log_level('ERROR')用来抑制日志生成
        with mne.use_log_level('ERROR'):

            raw = mne.io.RawArray(data.copy(), info)  # create raw
            raw = raw.filter(l_freq=1.0, h_freq=None)

            ica = ICA(
                n_components=n_components,
                method="fastica",
                fit_params=None,
                # max_iter = 800,
                max_iter="auto",
                random_state=0,
            )

            ica.fit(raw)

            # source代表的是对原始数据ica分析之后得到的成分序列
            # ica.plot_sources(raw, show_scrollbars=False)
            sources = ica.get_sources(raw)

            # print(sources)
            # print("acc:", (np.array(sources.get_data())==np.array(data)).sum()/(data.shape[0]*data.shape[1]))

            ica_components = sources.get_data()
            # print([np.mean(np.abs(component)) for component in ica_components])
            ''' 频域分析 '''
            # statictics, components_f, noise_f, noise_indx, max_kurt, kurt_list = plot_ICAfreq(ica_components, this_attack_freq, origin_freq, 250)
            components_f, extre_frequency, max_kurt, kurt_list = self.plot_ICAfreq(ica_components, 250)
            ''' 频域分析 '''

            Ks = []
            for j in range(len(ica_components)):
                corr_coefs = np.zeros((0, 16))
                for noise_templete in self.noise_wave_templetes:
                    corr_coef = np.abs(np.corrcoef(ica_components[j:j+1], noise_templete[:, :]))
                    corr_coefs = np.concatenate((corr_coefs, corr_coef[0:1, 1:]), axis=0)
                Ks.append(kurtosis(corr_coefs.reshape(-1)))
                # print(corr_coefs)


            indx = [m for m, n in enumerate(Ks) if n>=pearson_threshold]
            success = False
            if len(indx)>0:
                success = True
            cha_set = set()
            ''' 只有在频域上峰度属于离群值，时域上挑选出的通道才可被剔除 '''
            ''' 频域上的其他离群值若与 时域上选出成分的频率 一致，则该成分也可被剔除 '''
            if if_FD:
                if len(indx) != 0:
                    cha_set.add(indx[-1])
                    temp_freq = components_f[indx[-1]]
                    # 前三个成分是否也可以被剔除？从噪声与eeg信号平等的角度来说，各成分应该一视同仁
                    for cha in range(n_components-1, 0, -1):
                        if self.whether_same(temp_freq, components_f[cha]) and len(extre_frequency[cha]) != 0:
                            cha_set.add(cha)
            # 是否展示
            if if_show and len(cha_set) != 0:
                # print(p_kurt_list)
                print("生成indx集合: ", cha_set)
                # print("原始信号: ", targets[int(origin_labels)-1], "攻击频率: ", this_attack_freq)
                print("时域剔除通道: ", indx, "\t", max(Ks), "\n频域峰度: ", kurt_list)
                print("成分频率: ", components_f, "\n", "优质频率: ", extre_frequency, "\n")

            if if_FD:
                indx = list(cha_set)
                indx.sort()
            clean_ica = ica.copy()
            clean_ica.exclude = indx
            clean_data = clean_ica.apply(raw.copy())

            # 输出最高峰度通道(无用，无法找到被攻击通道)
        '''
        apply()方法的主要作用是:
        将ICA模型提取出的成分混合矩阵应用到数据上,进行源信号的重构。
        去除指定的成分,用于去除数据中的噪声成分。
        在apply时:
        如果不指定exclude,则对信号进行ICA分解重构,但不去除成分
        如果指定exclude,则会在重构时去除这些成分
        '''
        return clean_data.get_data(), success
    

    def ICA_noise_analysis(self, data, pearson_threshold=10, n_components=9):
        info = mne.create_info(
            ch_names=list(str(i) for i in range(data.shape[0])),
            ch_types="eeg",  # channel type
            sfreq=250
        )
        # use_log_level('ERROR')用来抑制日志生成
        with mne.use_log_level('ERROR'):

            raw = mne.io.RawArray(data.copy(), info)  # create raw
            raw = raw.filter(l_freq=1.0, h_freq=None)

            ica = ICA(
                n_components=n_components,
                method="fastica",
                fit_params=None,
                max_iter="auto",
                random_state=0,
            )

            ica.fit(raw)

            # source代表的是对原始数据ica分析之后得到的成分序列
            # ica.plot_sources(raw, show_scrollbars=False)
            sources = ica.get_sources(raw)

            # print(sources)
            # print("acc:", (np.array(sources.get_data())==np.array(data)).sum()/(data.shape[0]*data.shape[1]))

            ica_components = sources.get_data()
            
            ''' 时域分析 '''
            Ks = []
            for j in range(len(ica_components)):
                corr_coefs = np.zeros((0, 16))
                for noise_templete in self.noise_wave_templetes:
                    corr_coef = np.abs(np.corrcoef(ica_components[j:j+1], noise_templete[:, :]))
                    corr_coefs = np.concatenate((corr_coefs, corr_coef[0:1, 1:]), axis=0)
                Ks.append(kurtosis(corr_coefs.reshape(-1)))
            success = any(K > pearson_threshold for K in Ks)
            max_corrs = np.max(corr_coefs, axis=1)
            max_max_corrs = [np.max(max_corrs)]
            max_max_index = np.where(max_corrs == max_max_corrs)[0][0]
            other_max_corrs = np.delete(max_corrs, max_max_index).tolist()

        '''
        apply()方法的主要作用是:
        将ICA模型提取出的成分混合矩阵应用到数据上,进行源信号的重构。
        去除指定的成分,用于去除数据中的噪声成分。
        在apply时:
        如果不指定exclude,则对信号进行ICA分解重构,但不去除成分
        如果指定exclude,则会在重构时去除这些成分
        '''
        if success:
            return success, max_max_corrs, other_max_corrs
        else:
            return success, None, None