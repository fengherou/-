# step2_measurement.py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config
from step1_machining import calc_ideal_z, generate_machining_error, get_scene

def simulate_measurement():
    radii = np.arange(config.MEAS_GAP, config.R_MAX + config.MEAS_GAP, config.MEAS_GAP)
    angles = np.linspace(0, 2 * np.pi, config.MEAS_PTS_PER_REV, endpoint=False)
    R, Theta = np.meshgrid(radii, angles)
    x_m, y_m = (R * np.cos(Theta)).flatten(), (R * np.sin(Theta)).flatten()
    z_id = calc_ideal_z(x_m, y_m)
    err = generate_machining_error(x_m, y_m)
    
    eps = 1e-5
    Zx = (calc_ideal_z(x_m+eps, y_m) - calc_ideal_z(x_m-eps, y_m)) / (2*eps)
    Zy = (calc_ideal_z(x_m, y_m+eps) - calc_ideal_z(x_m, y_m-eps)) / (2*eps)
    c_err = config.PROBE_RADIUS * (1.0 - 1.0 / np.sqrt(Zx**2 + Zy**2 + 1.0))
    z_raw = (z_id + err) + c_err
    return x_m, y_m, z_raw, z_raw - c_err, z_id, radii

def plot_step2_scientific(x_m, y_m, z_raw, z_comp, z_id, radii):
    print("正在渲染 Step 2 图表...")
    fig = make_subplots(
        rows=3, cols=3,
        specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}],
               [{'type': 'scene'}, {'type': 'scene'}, {'type': 'xy'}],
               [{'type': 'table', 'colspan': 3}, None, None]],
        subplot_titles=('(a) 宏观 轮廓仪测量路径', '(b) 宏观 轮廓仪点云', '(c) 3D 测量误差分布热力图', 
                        '(d) 微观 路径提取误差', '(e) 微观 点云提取误差', '(f) 2D 误差对称性投影', ''),
        row_heights=[0.42, 0.42, 0.16], vertical_spacing=0.08, horizontal_spacing=0.03
    )
    
    grid_x, grid_y = np.meshgrid(np.linspace(-58, 58, 100), np.linspace(-58, 58, 100))
    grid_z_id, grid_z_0 = calc_ideal_z(grid_x, grid_y), np.zeros_like(grid_x)
    step = max(1, len(x_m) // 3000)
    xc, yc, err_nm = x_m[::step], y_m[::step], (z_comp[::step] - z_id[::step]) * 1e6

    # --- 列1: 测量路径 ---
    fig.add_trace(go.Surface(x=grid_x, y=grid_y, z=grid_z_id, colorscale='Greys', opacity=0.4, showscale=False), row=1, col=1)
    fig.add_trace(go.Surface(x=grid_x, y=grid_y, z=grid_z_0, colorscale='Greys', opacity=0.3, showscale=False), row=2, col=1)
    for r in radii[::3]: # 疏散圆环
        th = np.linspace(0, 2*np.pi, 180)
        _xc, _yc = r*np.cos(th), r*np.sin(th)
        _ec = generate_machining_error(_xc, _yc)*1e6
        fig.add_trace(go.Scatter3d(x=_xc, y=_yc, z=calc_ideal_z(_xc,_yc)+_ec*1e-6, mode='lines', line=dict(color='green', width=1.0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter3d(x=_xc, y=_yc, z=_ec, mode='lines', line=dict(color='green', width=1.0), showlegend=False), row=2, col=1)

    # --- 列2: 测量点云 ---
    fig.add_trace(go.Surface(x=grid_x, y=grid_y, z=grid_z_id, colorscale='Greys', opacity=0.4, showscale=False), row=1, col=2)
    fig.add_trace(go.Scatter3d(x=xc, y=yc, z=calc_ideal_z(xc,yc)+err_nm*1e-6, mode='markers', marker=dict(size=0.8, color='darkgreen', opacity=0.9), showlegend=False), row=1, col=2)
    fig.add_trace(go.Surface(x=grid_x, y=grid_y, z=grid_z_0, colorscale='Greys', opacity=0.3, showscale=False), row=2, col=2)
    fig.add_trace(go.Scatter3d(x=xc, y=yc, z=err_nm, mode='markers', marker=dict(size=0.8, color='darkgreen', opacity=0.9), showlegend=False), row=2, col=2)

    # --- 列3: 热力图 ---
    from scipy.interpolate import griddata
    grid_err = griddata((x_m, y_m), (z_comp - z_id)*1e6, (grid_x, grid_y), method='cubic')
    cb_args = dict(title="误差(nm)", orientation="h", x=0.835, y=-0.08, len=0.3, xanchor="center", yanchor="top")
    fig.add_trace(go.Surface(x=grid_x, y=grid_y, z=grid_err, colorscale='RdBu', showscale=False), row=1, col=3)
    fig.add_trace(go.Contour(z=grid_err, x=grid_x[0,:], y=grid_y[:,0], colorscale='RdBu', colorbar=cb_args), row=2, col=3)
    fig.update_xaxes(title_text="X (mm)", scaleanchor="y", scaleratio=1, row=2, col=3)
    fig.update_yaxes(title_text="Y (mm)", row=2, col=3)

    fig.add_trace(go.Table(header=dict(values=['<b>状态</b>', '<b>提取误差 RMS (nm)</b>', '<b>提取误差 PV (nm)</b>'], fill_color='darkblue', font=dict(color='white', size=14)),
                           cells=dict(values=[['仪器原数据', '软件补偿后'], [f"{np.std(z_raw-z_id)*1e6:.2f}", f"{np.std(z_comp-z_id)*1e6:.2f}"], [f"{(np.max(z_raw-z_id)-np.min(z_raw-z_id))*1e6:.2f}", f"{(np.max(z_comp-z_id)-np.min(z_comp-z_id))*1e6:.2f}"]], fill_color='aliceblue', height=30)), row=3, col=1)

    fig.update_layout(title="Step 2: 轮廓仪测量", width=2000, height=1400, template="plotly_white",
                      scene1=get_scene('Z(mm)'), scene2=get_scene('Z(mm)'), scene3=get_scene('误差 Z(nm)'),
                      scene4=get_scene('误差 Z(nm)'), scene5=get_scene('误差 Z(nm)'))
    fig.show()

if __name__ == "__main__":
    x_m, y_m, z_r, z_c, z_i, rads = simulate_measurement()
    plot_step2_scientific(x_m, y_m, z_r, z_c, z_i, rads)