# main_test.py
import config
from step1_machining import plot_step1_scientific, calc_ideal_z, generate_machining_error
from step2_measurement import simulate_measurement, plot_step2_scientific
from step3_fitting import fit_poly3, fit_zer, fit_poly5, plot_step3_scientific
from step4_compensation import plot_step4_scientific
import numpy as np

if __name__ == "__main__":
    print("超精密慢刀伺服 (STS) 自由曲面加工")

    # 执行 Step 1 
    print("[1/4] 正在执行 Step 1")
    theta = np.linspace(0, (config.R_MAX / config.FEED_RATE) * 2 * np.pi, 20000)
    r = (config.FEED_RATE / (2 * np.pi)) * theta
    x_t, y_t = r * np.cos(theta), r * np.sin(theta)
    plot_step1_scientific(x_t, y_t, calc_ideal_z(x_t, y_t), generate_machining_error(x_t, y_t))

    # 执行 Step 2 
    print("[2/4] 正在执行 Step 2")
    x_m, y_m, z_raw, z_comp, z_ideal, rads = simulate_measurement()
    plot_step2_scientific(x_m, y_m, z_raw, z_comp, z_ideal, rads)

    # ---------- 执行 Step 3 ----------
    print("[3/4] 正在执行 Step 3")
    err = z_comp - z_ideal
    c_p3 = fit_poly3(x_m, y_m, err)
    c_zer = fit_zer(x_m, y_m, err)
    c_p5 = fit_poly5(x_m, y_m, err)
    plot_step3_scientific(x_m, y_m, err, c_p3, c_zer, c_p5)

    # ---------- 执行 Step 4 & 5 ----------
    print("[4/4] 正在执行 Step 4")
    plot_step4_scientific(x_m, y_m, err, c_zer)

    print("\n所有仿真计算与渲染均已完成")