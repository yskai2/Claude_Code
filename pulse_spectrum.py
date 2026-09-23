"""
高斯脉冲激励 u0(t) 的频谱
=========================

    u0(t) = 0.09*0.0017453 * exp(-((t-t_pulse)/tau_pulse)^2)   [m]
    t_pulse   = 200e-6 s
    tau_pulse = 50e-6 s

画出：
    1. 时域波形 u0(t)
    2. 幅值谱 |U0(f)|（数值 FFT，并叠加解析高斯谱作为核对）

输出：pulse_time_and_spectrum.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================== 参数 ==============================

A0 = 0.09 * 0.0017453      # 脉冲幅值 [m]
t_pulse = 200e-6           # 脉冲中心时刻 [s]
tau_pulse = 50e-6          # 脉冲宽度参数 [s]

dt = 1e-6                  # 时间步长 [s]
N_t = 20001                 # 采样点数
t = np.arange(N_t) * dt     # 0 ... 20 ms

F_MAX = 30000.0             # 频谱绘图的频率上限 [Hz]

# ====================================================================


def u0(tt):
    return A0 * np.exp(-((tt - t_pulse) / tau_pulse) ** 2)


def u0_analytic_spectrum(f):
    """exp(-((t-t0)/tau)^2) 的解析傅里叶变换（连续时间，双边）：
       U0(f) = A0*tau*sqrt(pi) * exp(-(pi*f*tau)^2) * exp(-j*2*pi*f*t0)
       这里只用其幅值做数值 FFT 结果的核对曲线。"""
    return A0 * tau_pulse * np.sqrt(np.pi) * np.exp(-(np.pi * f * tau_pulse) ** 2)


if __name__ == "__main__":
    u = u0(t)

    # 数值 FFT（单边幅值谱，乘以 dt 使其逼近连续傅里叶变换的幅值）
    freq = np.fft.rfftfreq(N_t, d=dt)
    U_fft = np.fft.rfft(u) * dt
    U_mag = np.abs(U_fft)

    U_analytic = u0_analytic_spectrum(freq)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 时域波形
    ax1.plot(t * 1e6, u * 1e6, color="tab:blue")
    ax1.set_xlabel("Time (μs)")
    ax1.set_ylabel("u0(t) (μm)")
    ax1.set_title("Gaussian Pulse Excitation u0(t)")
    ax1.grid(True, alpha=0.3)

    # 频谱
    mask = freq <= F_MAX
    ax2.plot(freq[mask], U_mag[mask], color="tab:red", label="FFT (numerical)")
    ax2.plot(freq[mask], U_analytic[mask], "--", color="black",
              linewidth=1, label="Analytic Gaussian spectrum")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("|U0(f)| (m·s)")
    ax2.set_title("Amplitude Spectrum of u0(t)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("pulse_time_and_spectrum.png", dpi=150)
    plt.close(fig)
    print("Wrote pulse_time_and_spectrum.png")
