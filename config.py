
# config.py - 项目工艺参数配置文件

import numpy as np

# 1. 理想曲面方程参数 (双二次曲面)
Cx = 1 / 50000.0  # X方向曲率 单位毫米mm
Cy = 0.0          # Y方向曲率 (无穷大半径，倒数为0)
kx = 0.0          # X方向二次曲线常数
ky = 0.0          # Y方向二次曲线常数

# 2. 工件物理参数
DIAMETER = 116.0  # 光学元件口径 (mm)
R_MAX = DIAMETER / 2.0  # 最大半径 58.0 mm

# 3. 慢刀伺服(STS)真实仿真加工参数 (降采样版本，防止内存溢出)
FEED_RATE = 0.1   # 刀具径向进给量 (mm/rev) - 真实工业约 0.005，此处为仿真放大
PTS_PER_REV = 720 # 主轴每转采样的点数 (相当于每 0.5 度一个点)

# 4. Taylor Hobson 测量仿真参数 
MEAS_GAP = 1.0    # 同心圆间距 (mm)
MEAS_PTS_PER_REV = 360 # 每一圈的测量点数


# 5.Taylor Hobson 接触式蓝宝石测头半径 (mm)
PROBE_RADIUS = 2.0
