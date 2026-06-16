# app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
import config

# 导入底层核心模块
from step1_machining import calc_ideal_z, generate_machining_error
from step2_measurement import simulate_measurement
from step3_fitting import fit_poly3, eval_poly3, fit_zer, eval_zer, fit_poly5, eval_poly5

# === 新增：用于生成具体定量 LaTeX 方程的格式化函数 ===
def get_poly3_latex(c):
    terms = ["", "x", "y", "x^2", "xy", "y^2", "x^3", "x^2y", "xy^2", "y^3"]
    res = []
    for val, term in zip(c, terms):
        if term == "": res.append(f"{val:.3e}")
        else: res.append(f"{val:+.3e}{term}")
    return "E_{fit}(x,y) = " + " ".join(res)

def get_poly5_latex(c):
    terms = ["", "x", "y", "x^2", "xy", "y^2", "x^3", "x^2y", "xy^2", "y^3", 
             "x^4", "x^3y", "x^2y^2", "xy^3", "y^4", "x^5", "x^4y", "x^3y^2", "x^2y^3", "xy^4", "y^5"]
    res1, res2 = [], []
    for i, (val, term) in enumerate(zip(c, terms)):
        s = f"{val:.3e}" if term == "" else f"{val:+.3e}{term}"
        if i < 10: res1.append(s)
        else: res2.append(s)
    return "\\begin{aligned} E_{fit}(x,y) &= " + " ".join(res1) + " \\\\ &\\quad " + " ".join(res2) + " \\end{aligned}"

def get_zernike_latex(c):
    terms = ["", "r\\cos\\theta", "r\\sin\\theta", "(2r^2-1)", "r^2\\cos(2\\theta)", "r^2\\sin(2\\theta)", 
             "(3r^3-2r)\\cos\\theta", "(3r^3-2r)\\sin\\theta", "(6r^4-6r^2+1)"]
    res = []
    for val, term in zip(c, terms):
        if term == "": res.append(f"{val:.3e}")
        else: res.append(f"{val:+.3e}{term}")
    return "E_{fit}(r,\\theta) = " + " ".join(res)
# =================================================

# 1. 页面全局配置与 CSS 注入
st.set_page_config(
    page_title="慢刀伺服超精密车削可视化",
    layout="wide"
)

st.title("慢刀伺服超精密车削可视化")
st.markdown("""
<div style="background-color: #f8f9fa; padding: 15px; border-left: 5px solid #0056b3; border-radius: 5px; margin-bottom: 20px;">
    <strong>项目概况：</strong> 本系统模拟了超精密光学元件从 <b>首次慢刀伺服超精密车削加工 → 轮廓仪测量 → 误差面拟合 → 反向补偿自由曲面</b> 的全流程闭环。<br>
</div>
""", unsafe_allow_html=True)


# 2. 核心数据缓存计算 (一次计算，全局复用)
@st.cache_data
def load_and_compute_data():
    """仅在网页首次加载时生成所有仿真数据"""
    # 1. 加工物理数据 
    th_vis = np.linspace(0, 30 * 2 * np.pi, 3000)
    r_vis = (config.R_MAX / (30 * 2 * np.pi)) * th_vis
    x_path = r_vis * np.cos(th_vis)
    y_path = r_vis * np.sin(th_vis)
    z_path_ideal = calc_ideal_z(x_path, y_path)
    err_path_nm = generate_machining_error(x_path, y_path) * 1e6
    
    # 2. 测量物理数据 (同心圆抽样点阵)
    x_m, y_m, z_raw, z_comp, z_id, rads = simulate_measurement()
    err_cloud_nm = (z_comp - z_id) * 1e6
    
    # 3. 拟合算法系数
    err_cloud_mm = z_comp - z_id
    c_p3 = fit_poly3(x_m, y_m, err_cloud_mm)
    c_zer = fit_zer(x_m, y_m, err_cloud_mm)
    c_p5 = fit_poly5(x_m, y_m, err_cloud_mm)
    
    # 4. 全局统一的渲染网格
    grid_x, grid_y = np.meshgrid(np.linspace(-58, 58, 80), np.linspace(-58, 58, 80))
    grid_z_ideal = calc_ideal_z(grid_x, grid_y)
    
    return (x_path, y_path, z_path_ideal, err_path_nm, 
            x_m, y_m, z_id, err_cloud_nm, rads,
            c_p3, c_zer, c_p5, 
            grid_x, grid_y, grid_z_ideal)

# 数据解包
(x_path, y_path, z_path_ideal, err_path_nm, 
 x_m, y_m, z_id, err_cloud_nm, radii,
 c_p3, c_zer, c_p5, 
 grid_x, grid_y, grid_z_ideal) = load_and_compute_data()

fx, fy = grid_x.flatten(), grid_y.flatten()

# 3. 通用 Plotly 样式配置
macro_scene = dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)', aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.5))
micro_scene = dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='误差 Z (nm)', aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.5))
cb_style = dict(orientation="h", y=-0.2, x=0.5, xanchor="center", yanchor="top", title_side="bottom")

# 点云尺寸缩小
marker_size = 0.5 

# 4. 四大核心模块 (Tabs)
tab1, tab2, tab3, tab4 = st.tabs(["步骤1: 慢刀伺服加工", "步骤2: 轮廓仪测量", "步骤3: 误差方程拟合", "步骤4: 路径补偿"])


# TAB 1: 慢刀伺服加工仿真
with tab1:
    with st.container(border=True): 
        st.markdown("#### 加工控制台")
        c1, c2 = st.columns(2)
        with c1:
            plot_t1 = st.radio("选择加工形貌展现方式:", ["阿基米德加工路径", "高密度加工点云", "加工表面误差热力图"], horizontal=True, key="pt1")
        with c2:
            if plot_t1 in ["阿基米德加工路径", "高密度加工点云"]:
                view_t1 = st.radio("Z轴维度:", ["宏观 (全曲面形貌)", "微观 (仅展现误差起伏)"], horizontal=True, key="vm1")
            else:
                view_t1 = st.radio("热力图视角:", ["二维投影热力图", "三维空间热力图"], horizontal=True, key="hm1")

    fig1 = go.Figure()
    
    if plot_t1 == "阿基米德加工路径":
        if "宏观" in view_t1:
            fig1.add_trace(go.Scatter3d(x=x_path, y=y_path, z=z_path_ideal + err_path_nm*1e-6, mode='lines', line=dict(color='red', width=1.5), name="螺旋刀轨"))
            fig1.add_trace(go.Surface(x=grid_x, y=grid_y, z=grid_z_ideal, colorscale='Greys', opacity=0.4, showscale=False))
            fig1.update_layout(scene=macro_scene, title="宏观: 机床刀尖螺旋线绝对运动轨迹")
        else:
            fig1.add_trace(go.Scatter3d(x=x_path, y=y_path, z=err_path_nm, mode='lines', line=dict(color='red', width=1.5), name="误差轨迹"))
            fig1.update_layout(scene=micro_scene, title="微观: 展平后的纯加工误差波动")

    elif plot_t1 == "高密度加工点云":
        if "宏观" in view_t1:
            fig1.add_trace(go.Scatter3d(x=x_path, y=y_path, z=z_path_ideal + err_path_nm*1e-6, mode='markers', marker=dict(size=marker_size, color='darkred'), name="加工点"))
            fig1.add_trace(go.Surface(x=grid_x, y=grid_y, z=grid_z_ideal, colorscale='Greys', opacity=0.4, showscale=False))
            fig1.update_layout(scene=macro_scene, title="宏观: 离散化后的机床加工点阵")
        else:
            fig1.add_trace(go.Scatter3d(x=x_path, y=y_path, z=err_path_nm, mode='markers', marker=dict(size=marker_size, color='darkred'), name="加工点误差"))
            fig1.update_layout(scene=micro_scene, title="微观: 提取出的加工点误差 (纯色展示)")

    elif plot_t1 == "加工表面误差热力图":
        grid_err_nm_path = generate_machining_error(fx, fy).reshape(grid_x.shape) * 1e6
        
        cb = cb_style.copy(); cb['title'] = "加工误差 (nm)"
        if "二维" in view_t1:
            fig1.add_trace(go.Contour(z=grid_err_nm_path, x=grid_x[0,:], y=grid_y[:,0], colorscale='Jet', colorbar=cb))
            fig1.update_layout(xaxis_title="X (mm)", yaxis_title="Y (mm)", title="全表面二维误差分布")
        else:
            fig1.add_trace(go.Surface(x=grid_x, y=grid_y, z=grid_err_nm_path, colorscale='Jet', colorbar=cb))
            fig1.update_layout(scene=micro_scene, title="全表面三维误差起伏")
            
    fig1.update_layout(height=650, margin=dict(b=40))
    st.plotly_chart(fig1, use_container_width=True)


# TAB 2: 轮廓仪测量仿真
with tab2:
    with st.container(border=True):
        st.markdown("#### 轮廓仪数据提取控制台")
        c1, c2 = st.columns(2)
        with c1:
            plot_t2 = st.radio("选择测量仪器展现方式:", ["同心圆测量轨迹", "稀疏测量点云", "测量提取误差热力图"], horizontal=True, key="pt2")
        with c2:
            if plot_t2 in ["同心圆测量轨迹", "稀疏测量点云"]:
                view_t2 = st.radio("Z轴维度:", ["宏观 (全曲面形貌)", "微观 (仅展现误差起伏)"], horizontal=True, key="vm2")
            else:
                view_t2 = st.radio("热力图视角:", ["二维投影热力图", "三维空间热力图"], horizontal=True, key="hm2")

    fig2 = go.Figure()
    
    if plot_t2 == "同心圆测量轨迹":
        if "宏观" in view_t2:
            fig2.add_trace(go.Surface(x=grid_x, y=grid_y, z=grid_z_ideal, colorscale='Greys', opacity=0.4, showscale=False))
            fig2.update_layout(scene=macro_scene, title="宏观: Taylor Hobson 同心圆扫描动作")
        else:
            fig2.update_layout(scene=micro_scene, title="微观: 沿扫描轨迹提取的误差信号")
            
        for r in radii[::2]: # 隔圈画以保证美观
            th = np.linspace(0, 2*np.pi, 180)
            _xc, _yc = r*np.cos(th), r*np.sin(th)
            _err = generate_machining_error(_xc, _yc)*1e6
            if "宏观" in view_t2:
                fig2.add_trace(go.Scatter3d(x=_xc, y=_yc, z=calc_ideal_z(_xc,_yc) + _err*1e-6, mode='lines', line=dict(color='green', width=2), showlegend=False))
            else:
                fig2.add_trace(go.Scatter3d(x=_xc, y=_yc, z=_err, mode='lines', line=dict(color='green', width=2), showlegend=False))

    elif plot_t2 == "稀疏测量点云":
        step = 2
        if "宏观" in view_t2:
            fig2.add_trace(go.Scatter3d(x=x_m[::step], y=y_m[::step], z=z_id[::step] + err_cloud_nm[::step]*1e-6, mode='markers', marker=dict(size=marker_size, color='darkgreen'), name="测量点云"))
            fig2.add_trace(go.Surface(x=grid_x, y=grid_y, z=grid_z_ideal, colorscale='Greys', opacity=0.4, showscale=False))
            fig2.update_layout(scene=macro_scene, title="宏观: 轮廓仪记录的离散坐标点")
        else:
            fig2.add_trace(go.Scatter3d(x=x_m[::step], y=y_m[::step], z=err_cloud_nm[::step], mode='markers', marker=dict(size=marker_size, color='darkgreen'), name="测量点误差"))
            fig2.update_layout(scene=micro_scene, title="微观: 剥离出的稀疏测量误差 (纯色展示)")

    elif plot_t2 == "测量提取误差热力图":
        grid_err_nm_meas = griddata((x_m, y_m), err_cloud_nm, (grid_x, grid_y), method='cubic')
        cb = cb_style.copy(); cb['title'] = "测量提取误差 (nm)"
        if "二维" in view_t2:
            fig2.add_trace(go.Contour(z=grid_err_nm_meas, x=grid_x[0,:], y=grid_y[:,0], colorscale='Jet', colorbar=cb))
            fig2.update_layout(xaxis_title="X (mm)", yaxis_title="Y (mm)", title="基于稀疏数据的插值恢复")
        else:
            fig2.add_trace(go.Surface(x=grid_x, y=grid_y, z=grid_err_nm_meas, colorscale='Jet', colorbar=cb))
            fig2.update_layout(scene=micro_scene, title="基于稀疏数据的 3D 恢复")
            
    fig2.update_layout(height=650, margin=dict(b=40))
    st.plotly_chart(fig2, use_container_width=True)


# TAB 3: 误差特征拟合
with tab3:
    with st.container(border=True):
        st.markdown("#### 误差曲面拟合控制台")
        c1, c2 = st.columns([1, 2])
        with c1:
            fit_method_t3 = st.selectbox("选择拟合方法:", ["Zernike 正交多项式", "3阶全局多项式", "5阶全局多项式"], key="fit_t3")
        with c2:
            plot_t3 = st.radio("评估视角:", ["原始测点云与拟合曲面对比", "连续误差拟合曲面", "残差热力图 (评价拟合质量)"], horizontal=True, key="pt3")
            if plot_t3 == "残差热力图 (评价拟合质量)":
                view_t3 = st.radio("热力图维度:", ["二维投影热力图", "三维空间热力图"], horizontal=True, key="hm3")

    if "Zernike" in fit_method_t3:
        err_fit_grid_nm = eval_zer(fx, fy, c_zer).reshape(grid_x.shape) * 1e6
        err_fit_pts_nm = eval_zer(x_m, y_m, c_zer) * 1e6
    elif "3阶" in fit_method_t3:
        err_fit_grid_nm = eval_poly3(fx, fy, c_p3).reshape(grid_x.shape) * 1e6
        err_fit_pts_nm = eval_poly3(x_m, y_m, c_p3) * 1e6
    else:
        err_fit_grid_nm = eval_poly5(fx, fy, c_p5).reshape(grid_x.shape) * 1e6
        err_fit_pts_nm = eval_poly5(x_m, y_m, c_p5) * 1e6

    residual_nm = err_cloud_nm - err_fit_pts_nm
    fig3 = go.Figure()

    if plot_t3 == "原始测点云与拟合曲面对比":
        step = 2
        fig3.add_trace(go.Surface(x=grid_x, y=grid_y, z=err_fit_grid_nm, colorscale='Viridis', opacity=0.8, name="拟合面"))
        fig3.add_trace(go.Scatter3d(x=x_m[::step], y=y_m[::step], z=err_cloud_nm[::step], mode='markers', marker=dict(size=marker_size, color='darkred'), name="真实测点"))
        fig3.update_layout(scene=micro_scene, title="数学拟合面穿过真实物理测点云", height=600)
        
    elif plot_t3 == "连续误差拟合曲面":
        cb = cb_style.copy(); cb['title'] = "拟合高度 (nm)"
        fig3.add_trace(go.Surface(x=grid_x, y=grid_y, z=err_fit_grid_nm, colorscale='Viridis', colorbar=cb))
        fig3.update_layout(scene=micro_scene, title=f"利用 {fit_method_t3} 重建的连续误差面", height=600)

    elif plot_t3 == "残差热力图 (评价拟合质量)":
        grid_res_nm = griddata((x_m, y_m), residual_nm, (grid_x, grid_y), method='cubic')
        cb = cb_style.copy(); cb['title'] = "拟合残差 (nm)"
        if view_t3 == "二维投影热力图":
            fig3.add_trace(go.Contour(z=grid_res_nm, x=grid_x[0,:], y=grid_y[:,0], colorscale='RdBu', colorbar=cb))
            fig3.update_layout(xaxis_title="X (mm)", yaxis_title="Y (mm)", title=f"{fit_method_t3} - 残差分布 (数值越小说明越贴合)", height=600)
        else:
            fig3.add_trace(go.Surface(x=grid_x, y=grid_y, z=grid_res_nm, colorscale='RdBu', colorbar=cb))
            fig3.update_layout(scene=micro_scene, title=f"{fit_method_t3} - 3D 残差波动", height=600)

    st.plotly_chart(fig3, use_container_width=True)
    
    # === 新增建议2：定性与定量给出具体的误差拟合方程 ===
    if plot_t3 == "连续误差拟合曲面":
        st.markdown(f"**【{fit_method_t3}】具体定量误差拟合方程：**")
        if "Zernike" in fit_method_t3:
            st.markdown("*(注：下式中极坐标转换规则为 $r = \\frac{\\sqrt{x^2+y^2}}{58}$, $\\theta = \\arctan(\\frac{y}{x})$)*")
            st.latex(get_zernike_latex(c_zer))
        elif "3阶" in fit_method_t3:
            st.latex(get_poly3_latex(c_p3))
        elif "5阶" in fit_method_t3:
            st.latex(get_poly5_latex(c_p5))
            
    st.info(f"**{fit_method_t3} 拟合精度指标：**  残差 RMS = **{np.std(residual_nm):.2f} nm**  |  残差 PV = **{np.max(residual_nm) - np.min(residual_nm):.2f} nm**")


# TAB 4: 刀具路径补偿
with tab4:
    with st.container(border=True):
        st.markdown("#### 自由曲面补偿及补偿系数")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            fit_method_t4 = st.selectbox("拟合模型:", ["Zernike 正交多项式", "3阶全局多项式", "5阶全局多项式"], key="fit_t4")
        with c2:
            K = st.slider("误差补偿系数 (K):", min_value=0.0, max_value=1.5, value=0.8, step=0.05)
        with c3:
            view_t4 = st.radio("右侧热力图视角:", ["二维投影热力图", "三维空间热力图"], horizontal=True, key="hm4")

    # 计算全局网格的补偿数据
    if "Zernike" in fit_method_t4:
        err_fit_grid_m = eval_zer(fx, fy, c_zer).reshape(grid_x.shape)
        err_fit_path_m = eval_zer(x_path, y_path, c_zer)
    elif "3阶" in fit_method_t4:
        err_fit_grid_m = eval_poly3(fx, fy, c_p3).reshape(grid_x.shape)
        err_fit_path_m = eval_poly3(x_path, y_path, c_p3)
    else:
        err_fit_grid_m = eval_poly5(fx, fy, c_p5).reshape(grid_x.shape)
        err_fit_path_m = eval_poly5(x_path, y_path, c_p5)

    # 网格曲面
    z_comp_surface = grid_z_ideal - K * err_fit_grid_m
    # 加工路径指令 (宏观与微观)
    z_comp_path_macro = z_path_ideal - K * err_fit_path_m
    z_comp_path_micro_nm = - K * err_fit_path_m * 1e6
    
    np.random.seed(42)
    noise_grid = np.random.randn(*grid_x.shape) * 10 * 1e-6 
    pred_res_nm = ((err_fit_grid_m - K * err_fit_grid_m) + noise_grid) * 1e6

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(f"##### 1. 补偿后加工指令展示 (K={K})")
        
        # === 新增建议1：左侧五幅图切换面板 ===
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            plot_t4_left = st.radio("选择左侧展示图:", ["补偿自由曲面", "补偿后的加工路径", "补偿后的加工点云图"], key="pt4_l", label_visibility="collapsed")
        with c_sub2:
            if plot_t4_left != "补偿自由曲面":
                view_t4_left = st.radio("指令维度视角:", ["宏观", "微观"], horizontal=True, key="vt4_l", label_visibility="collapsed")
            else:
                view_t4_left = "宏观"

        fig4_left = go.Figure()
        
        if plot_t4_left == "补偿自由曲面":
            cb_path = cb_style.copy(); cb_path['title'] = "Z轴绝对坐标 (mm)"
            fig4_left.add_trace(go.Surface(x=grid_x, y=grid_y, z=z_comp_surface, colorscale='Blues', colorbar=cb_path))
            fig4_left.update_layout(scene=macro_scene, height=550, margin=dict(l=0, r=0, b=50, t=30))
            
        elif plot_t4_left == "补偿后的加工路径":
            if "宏观" in view_t4_left:
                fig4_left.add_trace(go.Scatter3d(x=x_path, y=y_path, z=z_comp_path_macro, mode='lines', line=dict(color='blue', width=1.5)))
                fig4_left.add_trace(go.Surface(x=grid_x, y=grid_y, z=grid_z_ideal, colorscale='Greys', opacity=0.4, showscale=False))
                fig4_left.update_layout(scene=macro_scene, height=550, margin=dict(l=0, r=0, b=50, t=30))
            else:
                fig4_left.add_trace(go.Scatter3d(x=x_path, y=y_path, z=z_comp_path_micro_nm, mode='lines', line=dict(color='blue', width=1.5)))
                fig4_left.update_layout(scene=micro_scene, height=550, margin=dict(l=0, r=0, b=50, t=30))
                
        elif plot_t4_left == "补偿后的加工点云图":
            if "宏观" in view_t4_left:
                fig4_left.add_trace(go.Scatter3d(x=x_path, y=y_path, z=z_comp_path_macro, mode='markers', marker=dict(size=marker_size, color='darkblue')))
                fig4_left.add_trace(go.Surface(x=grid_x, y=grid_y, z=grid_z_ideal, colorscale='Greys', opacity=0.4, showscale=False))
                fig4_left.update_layout(scene=macro_scene, height=550, margin=dict(l=0, r=0, b=50, t=30))
            else:
                fig4_left.add_trace(go.Scatter3d(x=x_path, y=y_path, z=z_comp_path_micro_nm, mode='markers', marker=dict(size=marker_size, color='darkblue')))
                fig4_left.update_layout(scene=micro_scene, height=550, margin=dict(l=0, r=0, b=50, t=30))
                
        st.plotly_chart(fig4_left, use_container_width=True)
        
        # 新增建议1附带：当显示补偿曲面时，给出明确的数学方程
        if plot_t4_left == "补偿自由曲面":
            st.markdown("**具体补偿后的三维二次曲面方程：**")
            ideal_eq = "\\frac{\\frac{1}{50000} x^2}{1 + \\sqrt{1 - (\\frac{1}{50000})^2 x^2}}"
            st.latex(f"Z_{{comp}}(x,y) = {ideal_eq} - {K} \\cdot E_{{fit}}(x,y)")

    with col_right:
        st.markdown(f"##### 2. 补偿后的加工误差")
        fig4_right = go.Figure()
        cb_res = cb_style.copy(); cb_res['title'] = "加工误差 (nm)"
        
        if view_t4 == "二维投影热力图":
            fig4_right.add_trace(go.Contour(z=pred_res_nm, x=grid_x[0,:], y=grid_y[:,0], colorscale='RdBu', colorbar=cb_res, zmin=-150, zmax=150))
            fig4_right.update_layout(xaxis_title="X (mm)", yaxis_title="Y (mm)", height=550, margin=dict(l=0, r=0, b=50, t=30))
        else: 
            fig4_right.add_trace(go.Surface(x=grid_x, y=grid_y, z=pred_res_nm, colorscale='RdBu', colorbar=cb_res, cmin=-150, cmax=150))
            fig4_right.update_layout(scene=micro_scene, height=550, margin=dict(l=0, r=0, b=50, t=30))
            
        st.plotly_chart(fig4_right, use_container_width=True)

    # 动态预警色块
    pred_rms = np.std(pred_res_nm)
    if K <= 0.6:
        st.warning(f"**欠补偿 (K={K}) ** 预计加工后收敛较慢，残余 RMS = **{pred_rms:.2f} nm**。")
    elif 0.6 < K <= 0.9:
        st.success(f"**最佳补偿 (K={K}) ** 能够抵消材料回弹且一定程度上避免过切，残余 RMS = **{pred_rms:.2f} nm**。")
    else:
        st.error(f"**过补偿 (K={K}) ** 极易导致不可逆的过切，残余 RMS = **{pred_rms:.2f} nm**。")