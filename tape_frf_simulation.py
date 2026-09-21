"""
磁带横向振动频率响应 (FRF) 仿真
================================

模型说明
--------
磁带横向振动被建模为受张力 T 作用的一根有阻尼弦（string under tension）：

    rho*A * d2y/dt2 + c * dy/dt - T * d2y/dx2 = 0,   0 <= x <= L

边界条件：
    x = 0 (出带端)  : y(0, t) = u0(t)   —— 施加高斯脉冲位移激励
    x = L (另一端)  : y(L, t) = 0       —— 固定（导辊约束）

响应输出点：磁带中点 x = L/2。

求解方法：模态叠加法 (modal superposition)。令
    y(x,t) = (1 - x/L) * u0(t) + w(x,t),   w(0,t) = w(L,t) = 0
把非齐次边界条件“吸收”进已知的静态型函数中，w(x,t) 满足零边界条件，
可用正弦模态展开 w(x,t) = sum_n q_n(t) * sin(n*pi*x/L) 求解，
每个模态坐标 q_n(t) 满足一个二阶阻尼振子方程（推导见下方代码注释），
用经典 RK4 做时域积分，效率高且不受显式差分法的空间 CFL 稳定性限制。

磁带/张力等物理参数为典型 1/2 英寸磁带 (如 LTO/DLT 量级) 的合理假设值，
在下方"用户可调参数"区域给出，可根据实际磁带/机构参数修改。

输出
----
1. output.csv：两列 (序号, 位移响应[m])，共 8*20001 = 160008 行。
   依次为张力 0.3N, 0.4N, ..., 1.0N 下磁带中点的位移响应时程（各20001行）。
2. tape_frf.png：8 条张力对应的频率响应函数(FRF)幅值谱，绘制在同一张图中。
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================== 用户可调参数 ==============================

# --- 磁带材料与几何参数（假设为典型 1/2 英寸磁带，可按实际情况修改）---
rho = 1390.0            # 磁带基膜密度 [kg/m^3]（PEN 基膜典型值）
tape_width = 12.65e-3   # 磁带宽度 [m]（1/2 英寸）
tape_thickness = 5.6e-6 # 磁带厚度 [m]
rhoA = rho * tape_width * tape_thickness   # 单位长度质量 [kg/m]

L = 0.15                # 磁带自由跨长（出带端到固定端）[m]
zeta = 0.02              # 模态阻尼比（假设各阶模态相同）
N_modes = 60             # 模态叠加所取的模态阶数

# --- 张力工况 ---
tensions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]   # [N]

# --- 高斯脉冲激励参数（施加于出带端 x=0）---
A0 = 0.09 * 0.0017453     # 脉冲幅值 [m]
t_pulse = 200e-6          # 脉冲中心时刻 [s]
tau_pulse = 50e-6         # 脉冲宽度参数 [s]

# --- 时间离散化 ---
dt = 1e-6                 # 时间步长 [s]
N_t = 20001                # 每个张力工况的采样点数（含 t=0）
t = np.arange(N_t) * dt    # 0 ... 20 ms

# ============================================================================


def u0(tt):
    """出带端高斯脉冲位移激励 u0(t) [m]"""
    return A0 * np.exp(-((tt - t_pulse) / tau_pulse) ** 2)


def u0_ddot(tt):
    """u0(t) 的二阶导数（解析表达式，用于模态力）"""
    x = (tt - t_pulse) / tau_pulse
    return A0 * np.exp(-x ** 2) * (4 * x ** 2 - 2) / tau_pulse ** 2


# 模态推导要点：
#   将 y(x,t) = (1-x/L)*u0(t) + w(x,t) 代入弦振动方程，w 用 sin(n*pi*x/L) 展开，
#   利用模态正交性投影，并采用模态阻尼比 zeta 代替分布式粘性阻尼，
#   最终每阶模态坐标 q_n(t) 满足：
#       q_n'' + 2*zeta*omega_n*q_n' + omega_n^2*q_n = -u0''(t) * 2/(n*pi)
#   其中 omega_n = (n*pi/L) * sqrt(T/rhoA)。
#   中点响应： y(L/2,t) = 0.5*u0(t) + sum_n q_n(t)*sin(n*pi/2)

n_arr = np.arange(1, N_modes + 1)
s_mid = np.sin(n_arr * np.pi / 2)     # 各阶模态在 x=L/2 处的取值 (+1,0,-1,0,...)
modal_force_coef = -2.0 / (n_arr * np.pi)   # 模态力系数（不含 u0''(t)）

n_tension = len(tensions)
T_arr = np.array(tensions)

# omega_n[i, n] : 第 i 个张力工况、第 n 阶模态的固有频率
omega_n = (n_arr[None, :] * np.pi / L) * np.sqrt(T_arr[:, None] / rhoA)


def modal_accel(Q, V, tt):
    """返回各模态加速度 q_n''(t)，Q,V 形状均为 (n_tension, N_modes)"""
    F = modal_force_coef[None, :] * u0_ddot(tt)   # (n_tension, N_modes)
    return -2 * zeta * omega_n * V - omega_n ** 2 * Q + F


def simulate_all_tensions():
    """对所有张力工况同时做时域仿真（向量化 RK4），返回 y_mid, 形状 (n_tension, N_t)"""
    Q = np.zeros((n_tension, N_modes))
    V = np.zeros((n_tension, N_modes))
    y_mid = np.zeros((n_tension, N_t))
    y_mid[:, 0] = 0.5 * u0(t[0]) + Q @ s_mid

    for i in range(N_t - 1):
        ti = t[i]

        k1Q = V
        k1V = modal_accel(Q, V, ti)

        k2Q = V + 0.5 * dt * k1V
        k2V = modal_accel(Q + 0.5 * dt * k1Q, V + 0.5 * dt * k1V, ti + 0.5 * dt)

        k3Q = V + 0.5 * dt * k2V
        k3V = modal_accel(Q + 0.5 * dt * k2Q, V + 0.5 * dt * k2V, ti + 0.5 * dt)

        k4Q = V + dt * k3V
        k4V = modal_accel(Q + dt * k3Q, V + dt * k3V, ti + dt)

        Q = Q + dt / 6.0 * (k1Q + 2 * k2Q + 2 * k3Q + k4Q)
        V = V + dt / 6.0 * (k1V + 2 * k2V + 2 * k3V + k4V)

        y_mid[:, i + 1] = 0.5 * u0(t[i + 1]) + Q @ s_mid

    return y_mid


def write_output_csv(y_mid, path="output.csv"):
    """按张力从小到大依次拼接各工况的位移响应，写入 output.csv（序号, 位移响应）"""
    flat = y_mid.reshape(-1)             # C-order: 先遍历张力，再遍历时间
    idx = np.arange(1, flat.size + 1)
    df = pd.DataFrame({"序号": idx, "位移响应": flat})
    df.to_csv(path, index=False)
    return path


def compute_frf(y_mid):
    """计算每个张力工况的频率响应函数 H(f) = Y_mid(f) / U0(f)"""
    freq = np.fft.rfftfreq(N_t, d=dt)
    U0_fft = np.fft.rfft(u0(t))
    H = np.zeros((n_tension, freq.size), dtype=complex)
    for i in range(n_tension):
        Y_fft = np.fft.rfft(y_mid[i, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            H[i, :] = Y_fft / U0_fft
    return freq, H


def plot_frf(freq, H, path="tape_frf.png", f_max=4000.0):
    """将 8 个张力对应的 FRF 幅值谱画在同一张图中"""
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
    print("Running modal-superposition time-domain simulation ...")
    y_mid = simulate_all_tensions()

    csv_path = write_output_csv(y_mid)
    print(f"Wrote {csv_path}: {y_mid.size} rows "
          f"({n_tension} tensions x {N_t} samples each)")

    freq, H = compute_frf(y_mid)
    png_path = plot_frf(freq, H)
    print(f"Wrote {png_path}")
