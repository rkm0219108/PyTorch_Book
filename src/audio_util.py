import torch
import torchaudio
import matplotlib.pyplot as plt
import os
import math
import IPython
from IPython.display import Audio, display
from typing import Optional, Tuple


# 取得一段語音的描述統計量
def print_stats(waveform: torch.Tensor, sample_rate: Optional[int] = None) -> None:
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
def plot_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    title: str = "Waveform",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
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
def plot_specgram(
    waveform: torch.Tensor, sample_rate: int, title: str = "Spectrogram", xlim: Optional[Tuple[float, float]] = None
) -> None:
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
def play_audio(waveform: torch.Tensor, sample_rate: int) -> None:
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("不支援超過雙聲道的音檔.")


# 取得檔案資訊
def inspect_file(path: str) -> None:
    print("-" * 10)
    print("Source:", path)
    print("-" * 10)
    print(f" - File size: {os.path.getsize(path)} bytes")
    print(f" - {torchaudio.info(path)}")
