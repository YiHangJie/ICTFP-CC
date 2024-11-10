import scipy.io
import scipy.signal as signal
import numpy as np
import pickle
import os
np.random.seed(2023)

root_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data", "Benchmark")
attack_root_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data", "Benchmark_attacked")
info = scipy.io.loadmat(os.path.join(root_path, "Freq_Phase.mat"))
stim_event_freq = info['freqs'][0]

def load_subject_data(path):
    f = scipy.io.loadmat(path)
    # print(f["data"].shape)
    channels = [53, 54, 55, 57, 58, 59, 61, 62, 63]
    channels = [i - 1 for i in channels]
    data = np.zeros((240, len(channels), 1375))
    pre_data = np.zeros((240, len(channels), 1500))
    label = np.zeros(240, dtype=int)
    for block in range(6):
        for target in range(40):
            data[block * 40 + target] = f["data"][channels, 125:, target, block]
            pre_data[block * 40 + target] = f["data"][channels, :, target, block]
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

def attack(raw_data, pre_data, scale=0.2, freq_fixture=False):
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
            attack_freq = np.random.choice(stim_event_freq)

        attack_channels_num = np.random.randint(2, 8, 1)[0]
        channels = list(range(9))
        attack_channels = sorted(np.random.choice(channels, attack_channels_num, False))
        # 偏移相位，攻击类型，攻击频率，攻击通道数量，攻击通道
        attack_info.append((offset_phase, attack_type, attack_freq, attack_channels_num, attack_channels))

        attack_signal = generate_attack_signal(attack_freq, cal_time=5.5, fs=250)[attack_type]
        attack_signal = np.concatenate((np.zeros(offset_phase), attack_signal))[:data[trial].shape[1]]
        for channel in attack_channels:
            data[trial][channel] = data[trial][channel] + stds[channel] * attack_signal

    return data, attack_info

def Benchmark_generation(noise_amp=0.1):
    for serial in range(1, 36):
        path = os.path.join(root_path, f"S{serial}.mat")
        print(path)
        data, pre_data, labels = load_subject_data(path)

        attacked_data, attack_info = attack(data, pre_data, noise_amp, False)  # attack_info 包括：偏移相位，攻击类型，攻击频率，攻击通道数量，攻击通道

        with open(os.path.join(attack_root_path, f"S{serial}_{noise_amp}.pkl"), "wb") as f:
            pickle.dump({"data": data, "labels": labels, "pre_data": pre_data, "attacked_data": attacked_data,  "attack_info": attack_info}, f)
        f.close()

def Benchmark_read(subjects=list(range(1, 2)), noise_amp=0):
    # print("loading Benchmark ......")
    data = np.zeros((0, 9, 1375))
    labels = np.zeros(0)
    for serial in subjects:
        path = os.path.join(root_path, f"S{serial}.mat")
        sub_data, _, sub_labels = load_subject_data(path)
        data = np.concatenate((data, sub_data), axis=0)
        labels = np.concatenate((labels, sub_labels), axis=0)
        print(f"\rS{serial} complete.", end="")
    print("\r", end="")
    return data, labels

def Benchmark_attacked_read(subjects=list(range(1, 2)), noise_amp=0.1):
    # print("loading attacked Benchmark ......")
    data = np.zeros((0, 9, 1375))
    labels = np.zeros(0)
    info = []
    for serial in subjects:
        with open(os.path.join(attack_root_path, f"S{serial}_{noise_amp}.pkl"), "rb") as f:
            file = pickle.load(f)
            sub_data = file['data']
            sub_labels = file['labels']
            pre_data = file['pre_data']
            attacked_data = file['attacked_data']
            attack_info = file['attack_info']
        f.close()

        data = np.concatenate((data, attacked_data), axis=0)
        labels = np.concatenate((labels, sub_labels), axis=0)
        info = info + attack_info

        print(f"\rS{serial} complete.", end="")
    print("\r", end="")
    return data, labels



if __name__ == "__main__":
    # for serial in range(1, 36):
    #     path = os.path.join(root_path, f"S{serial}.mat")
    #     print(path)
    #     data, pre_data, labels = load_subject_data(path)
    #     print(data.shape)
    #     print(pre_data.shape)
    #     print(labels.shape)

    #     attacked_data, attack_info = attack(data, pre_data, 0.1, False)  # attack_info 包括：偏移相位，攻击类型，攻击频率，攻击通道数量，攻击通道
    #     print(attacked_data.shape)
    # data, labels = Benchmark_read()
    # print(data.shape, labels.shape)
    Benchmark_attacked_read()