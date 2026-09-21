"""
磁带横向振动频率响应 (FRF) —— 从已有位移响应数据 (output1.csv) 计算并绘图
======================================================================

前提：磁带中点在 8 个张力工况下的时域位移响应已经计算好，保存在
output1.csv 中（与 output.csv 同样的格式）：
    两列：第一列为序号，第二列为位移响应 [m]
    共 8*20001 = 160008 行，按张力从小到大依次拼接：
        第 1~20001 行     : T = 0.3 N 的位移响应 y1
        第 20002~40002 行 : T = 0.4 N 的位移响应 y2
        ...
        第 140008~160008 行: T = 1.0 N 的位移响应 y8

本脚本不重新做时域仿真，只是：
    1. 读取 output1.csv 并按张力切分成 8 段时程；
    2. 用与激励脉冲一致的时间轴，计算出带端激励 u0(t) 的频谱 U0(f)；
    3. 对每个张力工况计算 FRF：H(f) = Y_mid(f) / U0(f)；
    4. 把 8 条 FRF 幅值谱画在同一张图中。

使用方法
--------
    python tape_frf_from_csv.py output1.csv

若不带参数运行，默认读取当前目录下的 output1.csv。

若你的时间步长 dt / 采样点数 N_t / 张力列表 / 脉冲参数与下面默认值不同，
请直接修改"用户可调参数"区域，保证与生成 output1.csv 时用的完全一致，
否则算出的频率轴和 FRF 会不准确。
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================== 用户可调参数 ==============================

# --- 张力工况（需与 output1.csv 中数据段的顺序一致）---
tensions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]   # [N]

# --- 每个张力工况的时间采样点数（需与生成 output1.csv 时一致）---
N_t = 20001

# --- 时间离散化（需与生成 output1.csv 时一致）---
dt = 1e-6                 # 时间步长 [s]
t = np.arange(N_t) * dt    # 0 ... (N_t-1)*dt

# --- 高斯脉冲激励参数（出带端 x=0 处施加，用于计算输入频谱 U0(f)）---
A0 = 0.09 * 0.0017453     # 脉冲幅值 [m]
t_pulse = 200e-6          # 脉冲中心时刻 [s]
tau_pulse = 50e-6         # 脉冲宽度参数 [s]

# --- FRF 绘图频率上限 ---
F_MAX = 4000.0             # [Hz]

# ============================================================================


def u0(tt):
    """出带端高斯脉冲位移激励 u0(t) [m]"""
    return A0 * np.exp(-((tt - t_pulse) / tau_pulse) ** 2)


def load_responses(csv_path):
    """读取 output1.csv，按张力切分成 (n_tension, N_t) 数组"""
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"{csv_path} 需要至少两列（序号, 位移响应），"
                          f"实际列数 = {df.shape[1]}")
    disp = df.iloc[:, 1].to_numpy(dtype=float)   # 第二列：位移响应

    n_tension = len(tensions)
    expected_rows = n_tension * N_t
    if disp.size != expected_rows:
        raise ValueError(
            f"{csv_path} 共 {disp.size} 行数据，"
            f"但按 {n_tension} 个张力 x {N_t} 个采样点预期应为 {expected_rows} 行。"
            f"请检查 tensions / N_t 设置是否与 output1.csv 一致。"
        )

    y_mid = disp.reshape(n_tension, N_t)   # 每行为一个张力工况的完整时程
    return y_mid


def compute_frf(y_mid):
    """计算每个张力工况的频率响应函数 H(f) = Y_mid(f) / U0(f)"""
    n_tension = y_mid.shape[0]
    freq = np.fft.rfftfreq(N_t, d=dt)
    U0_fft = np.fft.rfft(u0(t))
    H = np.zeros((n_tension, freq.size), dtype=complex)
    for i in range(n_tension):
        Y_fft = np.fft.rfft(y_mid[i, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            H[i, :] = Y_fft / U0_fft
    return freq, H


def plot_frf(freq, H, path="tape_frf.png", f_max=F_MAX):
    """将各张力对应的 FRF 幅值谱画在同一张图中"""
    n_tension = H.shape[0]
    mask = freq <= f_max
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, n_tension))
    for i, T in enumerate(tensions):
        mag_db = 20 * np.log10(np.abs(H[i, mask]) + 1e-30)
        ax.plot(freq[mask], mag_db, color=colors[i], label=f"T = {T:.1f} N")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|H(f)| (dB, ref 1 m/m)")
    ax.set_title("Tape Lateral Vibration FRF at Midpoint under Different Tensions")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "output1.csv"

    print(f"Loading displacement responses from {csv_path} ...")
    y_mid = load_responses(csv_path)
    print(f"Loaded {y_mid.shape[0]} tension cases x {y_mid.shape[1]} samples each")

    freq, H = compute_frf(y_mid)
    png_path = plot_frf(freq, H)
    print(f"Wrote {png_path}")
