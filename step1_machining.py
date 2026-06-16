# step1_machining.py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config

def calc_ideal_z(x, y):
    return (config.Cx * x**2 + config.Cy * y**2) / (1 + np.sqrt(1 - (1+config.kx)*(config.Cx*x)**2 - (1+config.ky)*(config.Cy*y)**2))

def generate_machining_error(x, y):
    r_norm = np.sqrt(x**2 + y**2) / config.R_MAX
    theta = np.arctan2(y, x)
    e_det = np.cos(2 * theta) + 0.5 * np.sin(8 * theta)
    e_det -= np.mean(e_det)
    np.random.seed(42)
    e_noise = np.random.randn(len(x)) - np.mean(np.random.randn(len(x)))
    low_w, high_w = 0.0, 2.0
    for _ in range(50):
        mid_w = (low_w + high_w) / 2.0
        err_raw = e_det + mid_w * e_noise
        if (np.max(err_raw) - np.min(err_raw)) / np.std(err_raw) < 5.50: low_w = mid_w
        else: high_w = mid_w
    return (e_det + mid_w * e_noise) * ((100 * 1e-6) / np.std(e_det + mid_w * e_noise))

def get_scene(z_title):
    return dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title=z_title,
                aspectmode='manual', aspectratio=dict(x=1.6, y=1.6, z=0.6))

def plot_step1_scientific(x_tool, y_tool, z_ideal, err_actual):
    print("正在渲染 Step 图表...")
    fig = make_subplots(
        rows=3, cols=3,
        specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}],
               [{'type': 'scene'}, {'type': 'scene'}, {'type': 'xy'}],
               [{'type': 'table', 'colspan': 3}, None, None]],
        subplot_titles=('(a) 宏观 加工路径 (Z: mm)', '(b) 宏观 离散点云 (Z: mm)', '(c) 3D 误差表面热力图 (Z: nm)',
                        '(d) 微观 路径误差 (Z: nm)', '(e) 微观 点云误差 (Z: nm)', '(f) 2D 误差投影热力图 (寻迹像散)',
                        ''),
        row_heights=[0.42, 0.42, 0.16], vertical_spacing=0.08, horizontal_spacing=0.03
    )
    
    grid_x, grid_y = np.meshgrid(np.linspace(-58, 58, 100), np.linspace(-58, 58, 100))
    grid_z_id, grid_z_0 = calc_ideal_z(grid_x, grid_y), np.zeros_like(grid_x)
    
    th_vis = np.linspace(0, 30 * 2 * np.pi, 2000)
    r_vis = (config.R_MAX / (30 * 2 * np.pi)) * th_vis
    x_path, y_path = r_vis * np.cos(th_vis), r_vis * np.sin(th_vis)
    err_path = generate_machining_error(x_path, y_path) * 1e6
    
    step = max(1, len(x_tool) // 3000)
    
    # ======== 关键修复点 1：消除降采样导致的放射状摩尔纹 ========
    # 我们保持与原代码完全相同的降采样点数，但将其均匀分布在 30 圈阿基米德螺旋线上
    num_pts = len(x_tool[::step])
    th_cloud = np.linspace(0, 30 * 2 * np.pi, num_pts)
    r_cloud = (config.R_MAX / (30 * 2 * np.pi)) * th_cloud
    x_cloud = r_cloud * np.cos(th_cloud)
    y_cloud = r_cloud * np.sin(th_cloud)
    err_cloud = generate_machining_error(x_cloud, y_cloud) * 1e6
    # ==============================================================

    # --- 列1: 路径 ---
    fig.add_trace(go.Surface(x=grid_x, y=grid_y, z=grid_z_id, colorscale='Blues', opacity=0.4, showscale=False), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=x_path, y=y_path, z=calc_ideal_z(x_path,y_path)+err_path*1e-6, mode='lines', line=dict(color='red', width=1.0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Surface(x=grid_x, y=grid_y, z=grid_z_0, colorscale='Greys', opacity=0.3, showscale=False), row=2, col=1)
    fig.add_trace(go.Scatter3d(x=x_path, y=y_path, z=err_path, mode='lines', line=dict(color='red', width=1.0), showlegend=False), row=2, col=1)

    # --- 列2: 点云 ---
    # 尺寸缩小至 0.5，颜色和点数未做任何更改
    fig.add_trace(go.Surface(x=grid_x, y=grid_y, z=grid_z_id, colorscale='Blues', opacity=0.4, showscale=False), row=1, col=2)
    fig.add_trace(go.Scatter3d(x=x_cloud, y=y_cloud, z=calc_ideal_z(x_cloud,y_cloud)+err_cloud*1e-6, mode='markers', marker=dict(size=0.5, color='darkred', opacity=0.9), showlegend=False), row=1, col=2)
    fig.add_trace(go.Surface(x=grid_x, y=grid_y, z=grid_z_0, colorscale='Greys', opacity=0.3, showscale=False), row=2, col=2)
    fig.add_trace(go.Scatter3d(x=x_cloud, y=y_cloud, z=err_cloud, mode='markers', marker=dict(size=0.5, color='darkred', opacity=0.9), showlegend=False), row=2, col=2)

    # --- 列3: 热力图 ---
    # ======== 关键修复点 2：避免插值发散，直接在致密网格上调用底层物理函数算误差 ========
    err_nm = err_actual * 1e6
    grid_err_nm = generate_machining_error(grid_x.flatten(), grid_y.flatten()).reshape(grid_x.shape) * 1e6
    
    cb_args = dict(title="误差(nm)", orientation="h", x=0.835, y=-0.08, len=0.3, xanchor="center", yanchor="top")
    fig.add_trace(go.Surface(x=grid_x, y=grid_y, z=grid_err_nm, colorscale='Jet', showscale=False), row=1, col=3)
    fig.add_trace(go.Contour(z=grid_err_nm, x=grid_x[0,:], y=grid_y[:,0], colorscale='Jet', colorbar=cb_args), row=2, col=3)
    fig.update_xaxes(title_text="X (mm)", scaleanchor="y", scaleratio=1, row=2, col=3)
    fig.update_yaxes(title_text="Y (mm)", row=2, col=3)

    # --- 底部表格 ---
    fig.add_trace(go.Table(
        header=dict(values=['<b>评价指标</b>', '<b>数值</b>', '<b>物理意义</b>'], fill_color='darkblue', font=dict(color='white', size=14)),
        cells=dict(values=[['总生成点云', '均方根误差 (RMS)', '峰谷误差 (PV)'], [f"{len(x_tool)} 点", f"{np.std(err_nm):.2f} nm", f"{np.max(err_nm) - np.min(err_nm):.2f} nm"], ['插补分辨率', '反映平整度', '反映突变极值']], fill_color='aliceblue', height=30)
    ), row=3, col=1)

    fig.update_layout(title="Step 1: 慢刀伺服加工仿真", width=2000, height=1400, template="plotly_white",
                      scene1=get_scene('Z(mm)'), scene2=get_scene('Z(mm)'), scene3=get_scene('误差 Z(nm)'),
                      scene4=get_scene('误差 Z(nm)'), scene5=get_scene('误差 Z(nm)'))
    fig.show()

if __name__ == "__main__":
    th = np.linspace(0, (config.R_MAX / config.FEED_RATE) * 2 * np.pi, 20000)
    r = (config.FEED_RATE / (2 * np.pi)) * th
    x_t, y_t = r * np.cos(th), r * np.sin(th)
    plot_step1_scientific(x_t, y_t, calc_ideal_z(x_t, y_t), generate_machining_error(x_t, y_t))