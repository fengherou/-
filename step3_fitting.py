# step3_fitting.py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config
from step2_measurement import simulate_measurement
from step1_machining import get_scene, calc_ideal_z  # ★ 修复：引入了 calc_ideal_z

def fit_poly3(x, y, z): return np.linalg.lstsq(np.column_stack([np.ones_like(x), x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3]), z, rcond=None)[0]
def eval_poly3(x, y, c): return np.column_stack([np.ones_like(x), x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3]).dot(c)
def fit_zer(x, y, z):
    r, t = np.sqrt(x**2 + y**2)/config.R_MAX, np.arctan2(y, x)
    return np.linalg.lstsq(np.column_stack([np.ones_like(r), r*np.cos(t), r*np.sin(t), 2*r**2-1, r**2*np.cos(2*t), r**2*np.sin(2*t), (3*r**3-2*r)*np.cos(t), (3*r**3-2*r)*np.sin(t), 6*r**4-6*r**2+1]), z, rcond=None)[0]
def eval_zer(x, y, c):
    r, t = np.sqrt(x**2 + y**2)/config.R_MAX, np.arctan2(y, x)
    return np.column_stack([np.ones_like(r), r*np.cos(t), r*np.sin(t), 2*r**2-1, r**2*np.cos(2*t), r**2*np.sin(2*t), (3*r**3-2*r)*np.cos(t), (3*r**3-2*r)*np.sin(t), 6*r**4-6*r**2+1]).dot(c)
def fit_poly5(x, y, z):
    A = np.column_stack([np.ones_like(x), x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3, x**4, x**3*y, x**2*y**2, x*y**3, y**4, x**5, x**4*y, x**3*y**2, x**2*y**3, x*y**4, y**5])
    return np.linalg.lstsq(A, z, rcond=None)[0]
def eval_poly5(x, y, c):
    A = np.column_stack([np.ones_like(x), x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3, x**4, x**3*y, x**2*y**2, x*y**3, y**4, x**5, x**4*y, x**3*y**2, x**2*y**3, x*y**4, y**5])
    return A.dot(c)

def calc_metrics(y_t, y_p):
    return np.std(y_t-y_p)*1e6, (np.max(y_t-y_p)-np.min(y_t-y_p))*1e6

def plot_step3_scientific(x_m, y_m, err_true, c_p3, c_zer, c_p5):
    print("正在渲染 Step 3 图表...")
    fig = make_subplots(
        rows=5, cols=3,
        specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}],
               [{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}],
               [{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}],
               [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}],
               [{'type': 'table', 'colspan': 3}, None, None]],
        subplot_titles=('(1) 宏观 3阶多项式', '(2) 宏观 9项Zernike', '(3) 宏观 5阶复杂多项式',
                        '(4) 微观 3阶拟合面', '(5) 微观 9项Zernike', '(6) 微观 5阶拟合面',
                        '(7) 3D 残差热力面', '(8) 3D 残差热力面', '(9) 3D 残差热力面',
                        '(10) 2D 残差投影', '(11) 2D 残差投影', '(12) 2D 残差投影', ''),
        row_heights=[0.22, 0.22, 0.22, 0.22, 0.12], vertical_spacing=0.04, horizontal_spacing=0.03
    )
    
    gx, gy = np.meshgrid(np.linspace(-58, 58, 80), np.linspace(-58, 58, 80))
    fx, fy = gx.flatten(), gy.flatten()
    z_id, z_0 = calc_ideal_z(gx, gy), np.zeros_like(gx)
    
    z_p3, z_zer, z_p5 = eval_poly3(fx, fy, c_p3).reshape(gx.shape)*1e6, eval_zer(fx, fy, c_zer).reshape(gx.shape)*1e6, eval_poly5(fx, fy, c_p5).reshape(gx.shape)*1e6
    xs, ys, zs = x_m[::5], y_m[::5], err_true[::5]*1e6

    from scipy.interpolate import griddata
    res_p3 = griddata((x_m, y_m), err_true*1e6 - eval_poly3(x_m, y_m, c_p3)*1e6, (gx, gy), method='cubic')
    res_zer = griddata((x_m, y_m), err_true*1e6 - eval_zer(x_m, y_m, c_zer)*1e6, (gx, gy), method='cubic')
    res_p5 = griddata((x_m, y_m), err_true*1e6 - eval_poly5(x_m, y_m, c_p5)*1e6, (gx, gy), method='cubic')

    x_anchors = [0.155, 0.5, 0.835]

    for c, fit, res, x_pos in zip([1, 2, 3], [z_p3, z_zer, z_p5], [res_p3, res_zer, res_p5], x_anchors):
        fig.add_trace(go.Surface(x=gx, y=gy, z=z_id+fit*1e-6, colorscale='Greys', opacity=0.8, showscale=False), row=1, col=c)
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=calc_ideal_z(xs,ys)+zs*1e-6, mode='markers', marker=dict(size=0.8, color='darkred', opacity=0.8), showlegend=False), row=1, col=c)
        fig.add_trace(go.Surface(x=gx, y=gy, z=z_0, colorscale='Greys', opacity=0.3, showscale=False), row=2, col=c)
        fig.add_trace(go.Surface(x=gx, y=gy, z=fit, colorscale='Viridis', opacity=0.8, showscale=False), row=2, col=c)
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='markers', marker=dict(size=0.8, color='darkred', opacity=0.8), showlegend=False), row=2, col=c)
        fig.add_trace(go.Surface(x=gx, y=gy, z=res, colorscale='RdBu', showscale=False), row=3, col=c)
        cb = dict(title="残差(nm)", orientation="h", x=x_pos, y=-0.05, len=0.28, xanchor="center", yanchor="top")
        fig.add_trace(go.Contour(z=res, x=gx[0,:], y=gy[:,0], colorscale='RdBu', colorbar=cb), row=4, col=c)
        fig.update_xaxes(title_text="X (mm)", scaleanchor=f"y{c+3}", scaleratio=1, row=4, col=c)
        fig.update_yaxes(title_text="Y (mm)", row=4, col=c)

    m3, mz, m5 = calc_metrics(err_true, eval_poly3(x_m, y_m, c_p3)), calc_metrics(err_true, eval_zer(x_m, y_m, c_zer)), calc_metrics(err_true, eval_poly5(x_m, y_m, c_p5))
    fig.add_trace(go.Table(header=dict(values=['<b>算法</b>', '<b>RMS (nm)</b>', '<b>PV (nm)</b>'], fill_color='darkblue', font=dict(color='white', size=14)),
                           cells=dict(values=[['3阶多项式', '9项Zernike', '5阶多项式'], [f"{m3[0]:.2f}", f"{mz[0]:.2f}", f"{m5[0]:.2f}"], [f"{m3[1]:.2f}", f"{mz[1]:.2f}", f"{m5[1]:.2f}"]], fill_color='aliceblue', height=30)), row=5, col=1)

    fig.update_layout(title="Step 3: 误差特征拟合", width=2200, height=2000, template="plotly_white",
                      scene1=get_scene('Z(mm)'), scene2=get_scene('Z(mm)'), scene3=get_scene('Z(mm)'),
                      scene4=get_scene('Z(nm)'), scene5=get_scene('Z(nm)'), scene6=get_scene('Z(nm)'),
                      scene7=get_scene('残差(nm)'), scene8=get_scene('残差(nm)'), scene9=get_scene('残差(nm)'))
    fig.show()

if __name__ == "__main__":
    x_m, y_m, _, z_c, z_i, _ = simulate_measurement()
    err = z_c - z_i
    plot_step3_scientific(x_m, y_m, err, fit_poly3(x_m, y_m, err), fit_zer(x_m, y_m, err), fit_poly5(x_m, y_m, err))