'''
对于S1~S15，事件发生前0.5s + 2s时间窗口 + after event 0.5s
对于S16~S70，时间发生前0.5s + 3s时间窗口 + after event 0.5s
'''
import scipy.io
import numpy as np
from scipy import signal
import pickle
import os
np.random.seed(2023)

root_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data", "BETA")
attack_root_path = os.path.join(os.path.abspath("."), "data", "BETA_attacked")

stim_event_frequency = [8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4, 11.6, 11.8,
             12.0, 12.2, 12.4, 12.6, 12.8, 13.0, 13.2, 13.4, 13.6, 13.8, 14.0, 14.2, 14.4, 14.6, 14.8, 15.0, 15.2, 15.4, 15.6, 15.8, 8.0, 8.2, 8.4]

sample_rate = 250

def load_subject_data(path, serial):
    f = scipy.io.loadmat(path)
    # print(f.keys())
    # print(f)
    # label = f["suppl_info"]
    # print(label)
    dat = f["data"][0][0][0]

    # 被试名，年龄，性别，通道信息（通道索引，角度，坐标，通道名），频率，初始相位，平均窄带信噪比矩阵，平均宽带信噪比矩阵，初始相位半径，采样率
    # info = f['data'][0][0][1][0][0]

    channels = [53, 54, 55, 57, 58, 59, 61, 62, 63]
    channels = [i - 1 for i in channels]

    data_len = 625 if serial<=15 else 875

    data = np.zeros((160, len(channels), data_len))
    pre_data = np.zeros((160, len(channels), data_len))
    label = np.zeros(160, dtype=int)
    for block in range(4):
        for target in range(40):
            data[block * 40 + target] = dat[channels, 125:, block, target]
            pre_data[block * 40 + target] = dat[channels, 125:, block, target]
            label[block * 40 + target] = int(target + 1)
    return data, pre_data, label


def generate_attack_signal(frequency, cal_time, fs):
    t = np.linspace(0, cal_time - 1 / fs, num=int(cal_time * fs))
    # 生成正弦信号
    sin_wave = np.sin(2 * np.pi * frequency * t)
    # 生成方波信号
    square_wave = np.where(np.mod(t * frequency, 1) < 0.5, 1, -1)
    # 生成矩形波信号
    duty_cycle = 0.2  # 占空比为20%
    rec_wave = np.where(np.mod(t * frequency, 1) < duty_cycle, 1, -1)
    # 生成锯齿波信号
    sawtooth_wave = 2 * (np.mod(t * frequency, 1) - 0.5)
    return [sin_wave, square_wave, rec_wave, sawtooth_wave]


def attack(raw_data, pre_data, scale=0.2, freq_fixture=False, serial=0):
    data = raw_data.copy()
    # attack_types = {0: "sin", 1: "square_wave", 2: "rec_wave", 3: "sawtooth_wave"}
    attack_info = []
    for trial in range(len(data)):
        stds = np.std(pre_data[trial], axis=1) * scale
        # print(stds)
        offset_phase = np.random.randint(10)
        attack_type = np.random.randint(4)

        # 选择攻击频率固定还是随机
        if not freq_fixture:
            attack_freq = np.random.uniform(8, 16, 1)[0]
        else:
            attack_freq = np.random.choice(stim_event_frequency)

        attack_channels_num = np.random.randint(2, 8, 1)[0]
        channels = list(range(9))
        attack_channels = sorted(np.random.choice(channels, attack_channels_num, False))
        # 偏移相位，攻击类型，攻击频率，攻击通道数量，攻击通道
        attack_info.append((offset_phase, attack_type, attack_freq, attack_channels_num, attack_channels))

        cal_t = 2.5 if serial<=15 else 3.5

        attack_signal = generate_attack_signal(attack_freq, cal_time=cal_t, fs=250)[attack_type]
        attack_signal = np.concatenate((np.zeros(offset_phase), attack_signal))[:data[trial].shape[1]]
        for channel in attack_channels:
            data[trial][channel] = data[trial][channel] + stds[channel] * attack_signal

    return data, attack_info

def Beta_generation(noise_amp=0.1):
    for serial in range(1, 71):
        path = os.path.join(root_path, f"S{serial}.mat")
        print(path)
        data, pre_data, labels = load_subject_data(path, serial)

        attacked_data, attack_info = attack(data, pre_data, noise_amp, False, serial)  # attack_info 包括：偏移相位，攻击类型，攻击频率，攻击通道数量，攻击通道

        with open(os.path.join(attack_root_path, f"S{serial}_{noise_amp}.pkl"), "wb") as f:
            pickle.dump({"data": data, "labels": labels, "pre_data": pre_data, "attacked_data": attacked_data,  "attack_info": attack_info}, f)
        f.close()

def Beta_read(subjects=list(range(1, 71)), noise_amp=0):
    # print("loading BETA ......")
    data = np.zeros((0, 9, 625))
    labels = np.zeros(0)
    for serial in subjects:
        path = os.path.join(root_path, f"S{serial}.mat")
        sub_data, _, sub_labels = load_subject_data(path, serial=serial)
        data = np.concatenate((data, sub_data[:, :, :625]), axis=0)
        labels = np.concatenate((labels, sub_labels), axis=0)
        print(f"\rS{serial} complete.", end="")
    print("\rdata loading complete!")
    return data, labels

def Beta_attacked_read(subjects=list(range(1, 71)), noise_amp=0.1):
    # print("loading attacked Benchmark ......")
    data = np.zeros((0, 9, 625))
    labels = np.zeros(0)
    info = []
    for serial in subjects:
        with open(os.path.join(attack_root_path, f"S{serial}_{noise_amp}.pkl"), "rb") as f:
            file = pickle.load(f)
            sub_data = file['data']
            sub_labels = file['labels']
            pre_data = file['pre_data']
            attacked_data = file['attacked_data'][:, :, :625]
            attack_info = file['attack_info']
        f.close()

        data = np.concatenate((data, attacked_data), axis=0)
        labels = np.concatenate((labels, sub_labels), axis=0)
        info = info + attack_info

        print(f"\rS{serial} complete.", end="")
    print("\rdata loading complete!")
    return data, labels


if __name__ == "__main__":
    # Beta_read()
    Beta_attacked_read()