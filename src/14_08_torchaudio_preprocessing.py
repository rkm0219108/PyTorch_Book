#!/usr/bin/env python
# coding: utf-8

# # torchaudio 音檔解析

# ## 載入相關套件

# In[1]:


import torch
import torchaudio
import IPython
from IPython.display import Audio
import matplotlib.pyplot as plt
import os
import math

# ## 取得音檔

# In[2]:


import requests

path = "./audio/steam-train-whistle-daniel_simon.wav"
url = "https://pytorch-tutorial-assets.s3.amazonaws.com/steam-train-whistle-daniel_simon.wav"
with open(path, 'wb') as file_:
    file_.write(requests.get(url).content)

# ## 取得音檔的屬性(metadata)

# In[3]:


# 檔案來源：https://pytorch-tutorial-assets.s3.amazonaws.com/steam-train-whistle-daniel_simon.wav
wav_file = './audio/steam-train-whistle-daniel_simon.wav'

metadata = torchaudio.info(wav_file)
print(metadata)

# ## 播放音檔(wav)

# In[4]:


# autoplay=True：自動播放，不須按 PLAY 鍵
IPython.display.Audio(wav_file, autoplay=False)

# ## 定義操作音檔相關的函數

# In[5]:


# 取得一段語音的描述統計量
def print_stats(waveform, sample_rate=None):
    if sample_rate:
        print("Sample Rate:", sample_rate)
    print("維度:", tuple(waveform.shape))
    print("資料型態:", waveform.dtype)
    print(f" - 最大值:         {waveform.max().item():6.3f}")
    print(f" - 最小值:         {waveform.min().item():6.3f}")
    print(f" - 平均數:        {waveform.mean().item():6.3f}")
    print(f" - 標準差: {waveform.std().item():6.3f}")
    print()
    print(waveform)
    print()


# 繪製語音的波形
def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)


# 繪製語音的頻譜
def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)


# 播放語音
def play_audio(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        IPython.display.display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        IPython.display.display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("不支援超過雙聲道的音檔.")


# 取得檔案資訊
def inspect_file(path):
    print("-" * 10)
    print("Source:", path)
    print("-" * 10)
    print(f" - File size: {os.path.getsize(path)} bytes")
    print(f" - {torchaudio.info(path)}")


# ## 顯示語音的描述統計量

# In[6]:


waveform, sample_rate = torchaudio.load(wav_file)
print_stats(waveform, sample_rate=sample_rate)

# ## 顯示波形

# In[7]:


plot_waveform(waveform, sample_rate)

# ## 繪製頻譜

# In[8]:


plot_specgram(waveform, sample_rate)

# ## 存檔

# In[9]:


# 以 16-bit signed integer Linear PCM 編碼存檔
path = "./audio/PCM_S16.wav"
torchaudio.save(path, waveform, sample_rate, encoding="PCM_S", bits_per_sample=16)
inspect_file(path)

# In[10]:


path = "./audio/steam-train-whistle-daniel_simon.wav"
inspect_file(path)

# ## 重抽樣

# In[11]:


import torchaudio.functional as F

# 重抽樣率
resample_rate = 4000
resampled_waveform = F.resample(waveform, sample_rate, resample_rate)

# 繪製頻譜
plot_specgram(resampled_waveform, resample_rate)

# In[12]:


plot_waveform(resampled_waveform, resample_rate)

# In[13]:


path = "./audio/resample.wav"
torchaudio.save(path, resampled_waveform, resample_rate)

# autoplay=True：自動播放，不須按 PLAY 鍵
IPython.display.Audio(wav_file, autoplay=False)

# ## Data Augmentation

# In[14]:


import sox

# create transformer
tfm = sox.Transformer()

# 裁剪原音檔 5 至 10.5 秒的片段.
# tfm.trim(5, 10.5)

# 壓縮
tfm.compand()

# 應用 fade in/fade out 效果
tfm.fade(fade_in_len=1.0, fade_out_len=0.5)

# 產生輸出檔
path = "audio/steam-train-whistle-daniel_simon.wav"
out_path = "audio/test.wav"  # path.split('.')[0]+'.aiff'
if os.path.exists(out_path):
    os.remove(out_path)
tfm.build_file(path, out_path)

# 輸出至記憶體
array_out = tfm.build_array(input_filepath=path)

# 顯示應用的效果
tfm.effects_log

# ## 播放音檔(wav)

# In[15]:


# autoplay=True：自動播放，不須按 PLAY 鍵
IPython.display.Audio(out_path, autoplay=False)

# ## 特徵萃取(Feature Extraction)

# In[23]:


import torchaudio.functional as F
import torchaudio.transforms as T
import librosa

wav_file = './audio/speech.wav'
waveform, sample_rate = torchaudio.load(wav_file)

# In[17]:


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


# In[18]:


n_fft = 1024
win_length = None
hop_length = 512

# 時頻轉換定義
spectrogram = T.Spectrogram(
    n_fft=n_fft,  # 快速傅立葉轉換的長度(Size of FFT)
    win_length=win_length,  # 視窗大小(Window size)
    hop_length=hop_length,  # 視窗終非重疊的Hop length)
    center=True,  # 是否在音訊前後補資料，使t時間點的框居中
    pad_mode="reflect",  # 補資料的方式
    power=2.0,  # 時頻大小的指數(Exponent for the magnitude spectrogram)
)
# 進行時頻轉換
spec = spectrogram(waveform)

print_stats(spec)
plot_spectrogram(spec[0], title='torchaudio')

# ## GriffinLim：由頻譜還原為原始音訊

# In[19]:


# 原始音訊
plot_waveform(waveform, sample_rate, title="Original")

griffin_lim = T.GriffinLim(
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
)
waveform = griffin_lim(spec)

# 由頻譜還原後的音訊
plot_waveform(waveform, sample_rate, title="Reconstructed")

# ## 取得FBank

# In[20]:


# FBank 繪圖
def plot_mel_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Filter bank')
    axs.imshow(fbank, aspect='auto')
    axs.set_ylabel('frequency bin')
    axs.set_xlabel('mel bin')
    plt.show(block=False)


n_fft = 256
n_mels = 64
sample_rate = 6000

mel_filters = F.melscale_fbanks(
    int(n_fft // 2 + 1),  # 分成的組數(Number of frequencies to highlight)
    n_mels=n_mels,  # FBank 個數
    f_min=0.0,  # 最小的頻率
    f_max=sample_rate / 2.0,  # 最大的頻率
    sample_rate=sample_rate,  # 取樣率
    norm='slaney',  # 區域常態化(Area normalization)
)
plot_mel_fbank(mel_filters, "Mel Filter Bank - torchaudio")

# ## 取得 梅爾頻譜(MelSpectrogram)

# In[21]:


n_fft = 1024
win_length = None
hop_length = 512
n_mels = 128

mel_spectrogram = T.MelSpectrogram(
    sample_rate=sample_rate,  # 取樣率
    n_fft=n_fft,  # 快速傅立葉轉換的長度(Size of FFT)
    win_length=win_length,  # 視窗大小(Window size)
    hop_length=hop_length,  # 視窗終非重疊的Hop length)
    center=True,  # 是否在音訊前後補資料，使t時間點的框居中
    pad_mode="reflect",  # 補資料的方式
    power=2.0,  # 時頻大小的指數(Exponent for the magnitude spectrogram)
    norm='slaney',  # 區域常態化(Area normalization)
    n_mels=n_mels,  # FBank 個數
    mel_scale="htk",  # htk or slaney
)

melspec = mel_spectrogram(waveform)
plot_spectrogram(melspec[0], title="MelSpectrogram - torchaudio", ylabel='mel freq')

# ## 取得 梅爾倒頻譜(MFCC)

# In[28]:


n_fft = 2048
win_length = None
hop_length = 512
n_mels = 256
n_mfcc = 256

mfcc_transform = T.MFCC(
    sample_rate=sample_rate,
    n_mfcc=n_mfcc,  # MFCC 個數
    melkwargs={
        'n_fft': n_fft,
        'n_mels': n_mels,
        'hop_length': hop_length,
        'mel_scale': 'htk',
    },
)

mfcc = mfcc_transform(waveform)

plot_spectrogram(mfcc[0])

# ## 取得音高(Pitch)

# In[26]:


def plot_pitch(waveform, sample_rate, pitch):
    figure, axis = plt.subplots(1, 1)
    axis.set_title("Pitch Feature")
    axis.grid(True)

    end_time = waveform.shape[1] / sample_rate
    time_axis = torch.linspace(0, end_time, waveform.shape[1])
    axis.plot(time_axis, waveform[0], linewidth=1, color='gray', alpha=0.3)

    axis2 = axis.twinx()
    time_axis = torch.linspace(0, end_time, pitch.shape[1])
    ln2 = axis2.plot(time_axis, pitch[0], linewidth=2, label='Pitch', color='green')

    axis2.legend(loc=0)
    plt.show(block=False)


# 偵測音高
pitch = F.detect_pitch_frequency(waveform, sample_rate)

# 繪製音高
plot_pitch(waveform, sample_rate, pitch)

# # 特徵增補(Feature Augmentation)

# ## 音訊延伸轉換(Time Stretch)

# In[27]:


n_fft = 400
win_length = None
hop_length = None

# 時頻轉換定義
spectrogram = T.Spectrogram(
    n_fft=n_fft,  # 快速傅立葉轉換的長度(Size of FFT)
    win_length=win_length,  # 視窗大小(Window size)
    hop_length=hop_length,  # 視窗終非重疊的Hop length)
    center=True,  # 是否在音訊前後補資料，使t時間點的框居中
    pad_mode="reflect",  # 補資料的方式
    power=None,  # 時頻大小的指數(Exponent for the magnitude spectrogram)
)
# 進行時頻轉換
spec = spectrogram(waveform)

# In[52]:


# 音訊延伸轉換
stretch = T.TimeStretch()

# 音訊拉長1.2倍
rate = 1.2
spec_ = stretch(spec, rate)
plot_spectrogram(torch.abs(spec_[0]), title=f"Stretched x{rate}", aspect='equal', xmax=304)

plot_spectrogram(torch.abs(spec[0]), title="Original", aspect='equal', xmax=304)

# 音訊縮短0.9倍
rate = 0.9
spec_ = stretch(spec, rate)
plot_spectrogram(torch.abs(spec_[0]), title=f"Stretched x{rate}", aspect='equal', xmax=304)

# ## 時間遮罩(Time Masking)

# In[11]:


spec = spectrogram(waveform)

# In[12]:


spec.dtype

# In[13]:


torch.random.manual_seed(4)

plot_spectrogram(spec[0], title="Original")

masking = T.TimeMasking(time_mask_param=40)
spec2 = masking(spec)

plot_spectrogram(spec2[0], title="Masked along time axis")

# ## 頻率遮罩(Frequency Masking)

# In[20]:


torch.random.manual_seed(4)
plot_spectrogram(spec[0], title="Original")

masking = T.FrequencyMasking(freq_mask_param=80)
spec2 = masking(spec)

plot_spectrogram(spec2[0], title="Masked along frequency axis")

# In[ ]:
