import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import logging
import os

logger = logging.getLogger(__name__)

def plot_biomechanics(all_biomechanics, env):
    """
    Processes and plots biomechanics data collected over multiple evaluation runs into a single dashboard.
    
    Args:
        all_biomechanics (list of lists): [episode][timestep][dict of metrics]
        env: The environment instance, used to extract joint and actuator indices.
    """
    logger.info(f"Processing biomechanics data for {len(all_biomechanics)} episodes...")
    
    # 1. Truncate to min episode length
    min_len = min(len(ep) for ep in all_biomechanics)
    if min_len == 0:
        logger.warning("No biomechanics data to plot (empty episodes).")
        return
        
    num_eps = len(all_biomechanics)
    logger.info(f"Analyzing up to {min_len} timesteps across {num_eps} runs.")

    # 2. Extract Indices (Exact Match for robustness)
    def get_jnt_id(name):
        try:
            return env.mj_model.joint(name).id
        except ValueError:
            return -1

    def get_act_id(name):
        try:
            return env.mj_model.actuator(name).id
        except ValueError:
            return -1

    # Joint qpos addresses
    joint_names = {
        "Hip Flexion R": "hip_flexion_r",
        "Knee R (Prosthetic)": "osl_knee_angle_r",
        "Ankle R (Prosthetic)": "osl_ankle_angle_r",
        "Hip Flexion L": "hip_flexion_l",
        "Knee L": "knee_angle_l",
        "Ankle L": "ankle_angle_l"
    }
    joint_qpos_indices = {display: env.mj_model.jnt_qposadr[get_jnt_id(name)] for display, name in joint_names.items() if get_jnt_id(name) != -1}
    
    # Actuator Indices
    soleus_l_idx = get_act_id("soleus_l")
    knee_act_idx = get_act_id("osl_knee_torque_actuator")
    ankle_act_idx = get_act_id("osl_ankle_torque_actuator")
    
    # DOF Indices (for Moments)
    knee_l_dof = env.mj_model.joint("knee_angle_l").dofadr[0] if get_jnt_id("knee_angle_l") != -1 else -1
    knee_r_dof = env.mj_model.joint("osl_knee_angle_r").dofadr[0] if get_jnt_id("osl_knee_angle_r") != -1 else -1
    
    # Gears
    gear_knee = env.mj_model.actuator_gear[knee_act_idx, 0] if knee_act_idx != -1 else 1.0
    gear_ankle = env.mj_model.actuator_gear[ankle_act_idx, 0] if ankle_act_idx != -1 else 1.0

    # 3. Pre-allocate Data
    soleus_act = np.zeros((num_eps, min_len))
    knee_torque_rt = np.zeros((num_eps, min_len))
    ankle_torque_rt = np.zeros((num_eps, min_len))
    knee_l_moment = np.zeros((num_eps, min_len))
    knee_r_moment = np.zeros((num_eps, min_len))
    
    joint_angles = {name: np.zeros((num_eps, min_len)) for name in joint_qpos_indices.keys()}
    foot_heights = {k: np.zeros((num_eps, min_len)) for k in ["right_foot", "left_foot", "right_ankle", "left_ankle"]}
    
    imp_keys = ["knee_K", "knee_B", "knee_target", "ankle_K", "ankle_B", "ankle_target"]
    impedance_data = {k: np.zeros((num_eps, min_len)) for k in imp_keys}
    has_impedance = False

    # 4. Extract Steps
    for ep_idx in range(num_eps):
        for t in range(min_len):
            step_data = all_biomechanics[ep_idx][t]
            
            if soleus_l_idx != -1:
                soleus_act[ep_idx, t] = step_data["ctrl"][soleus_l_idx]
                
            if knee_act_idx != -1:
                knee_torque_rt[ep_idx, t] = step_data["actuator_force"][knee_act_idx] * gear_knee
            if ankle_act_idx != -1:
                ankle_torque_rt[ep_idx, t] = step_data["actuator_force"][ankle_act_idx] * gear_ankle
                
            if "qfrc_actuator" in step_data:
                if knee_l_dof != -1: knee_l_moment[ep_idx, t] = step_data["qfrc_actuator"][knee_l_dof]
                if knee_r_dof != -1: knee_r_moment[ep_idx, t] = step_data["qfrc_actuator"][knee_r_dof]
            
            if "heights" in step_data:
                for k in foot_heights.keys():
                    foot_heights[k][ep_idx, t] = step_data["heights"].get(k, 0.0)
                    
            for name, qidx in joint_qpos_indices.items():
                joint_angles[name][ep_idx, t] = step_data["qpos"][qidx]

            if "impedance" in step_data and step_data["impedance"]:
                has_impedance = True
                for k in imp_keys:
                    impedance_data[k][ep_idx, t] = step_data["impedance"].get(k, 0.0)

    # 5. Build Subplots (Dashboard)
    rows = 6 if has_impedance else 5
    subplot_titles = (
        "Muscle Activations & Actuator Torques", "",
        "Joint-Level Net Moments (Nm)", "",
        "Leg Kinematics (Left)", "Leg Kinematics (Right)",
        "Ground Clearance (Feet)", "Ground Clearance (Ankles)"
    )
    if has_impedance:
        subplot_titles += ("Impedance: Knee Gains", "Impedance: Ankle Gains")

    fig = make_subplots(
        rows=rows, cols=2,
        subplot_titles=subplot_titles,
        vertical_spacing=0.05,
        horizontal_spacing=0.08
    )

    def add_shaded_trace(fig, data, name, row, col, color="blue", y_title=""):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        x = np.arange(len(mean))
        fig.add_trace(go.Scatter(x=list(x)+list(x)[::-1], y=list(mean+std)+list(mean-std)[::-1],
                                 fill='toself', fillcolor=f'rgba({color}, 0.2)', line=dict(color='rgba(255,255,255,0)'),
                                 showlegend=False, name=f"{name} StdDev"), row=row, col=col)
        fig.add_trace(go.Scatter(x=x, y=mean, line=dict(color=f'rgb({color})'), name=f"{name} Mean"), row=row, col=col)
        fig.update_yaxes(title_text=y_title, row=row, col=col)

    # Row 1: Activations & Torques
    add_shaded_trace(fig, soleus_act, "Soleus L", 1, 1, "255,0,0", "Activation")
    add_shaded_trace(fig, knee_torque_rt, "Knee Torque R", 1, 2, "0,100,255", "Nm")

    # Row 2: Net Moments
    add_shaded_trace(fig, knee_l_moment, "Net Moment L", 2, 1, "100,100,100", "Nm")
    add_shaded_trace(fig, knee_r_moment, "Net Moment R", 2, 2, "0,150,255", "Nm")

    # Row 3: Kinematics L
    add_shaded_trace(fig, joint_angles["Knee L"], "Knee L", 3, 1, "150,0,250", "Rad")
    add_shaded_trace(fig, joint_angles["Ankle L"], "Ankle L", 3, 1, "100,0,200", "Rad")
    # Row 3: Kinematics R
    add_shaded_trace(fig, joint_angles["Knee R (Prosthetic)"], "Knee R", 3, 2, "255,100,0", "Rad")
    add_shaded_trace(fig, joint_angles["Ankle R (Prosthetic)"], "Ankle R", 3, 2, "200,80,0", "Rad")

    # Row 4: Foot Heights
    add_shaded_trace(fig, foot_heights["left_foot"], "Foot L", 4, 1, "0,150,0", "Z-Pos (m)")
    add_shaded_trace(fig, foot_heights["right_foot"], "Foot R", 4, 1, "0,200,0", "Z-Pos (m)")
    # Row 4: Ankle Heights
    add_shaded_trace(fig, foot_heights["left_ankle"], "Ankle L", 4, 2, "150,150,0", "Z-Pos (m)")
    add_shaded_trace(fig, foot_heights["right_ankle"], "Ankle R", 4, 2, "200,200,0", "Z-Pos (m)")
    # Add ground line
    for c in [1, 2]: fig.add_shape(type="line", x0=0, y0=0, x1=min_len, y1=0, line=dict(color="black", dash="dash"), row=4, col=c)

    if has_impedance:
        # Row 5/6: Impedance
        add_shaded_trace(fig, impedance_data["knee_K"], "Knee K", 5, 1, "0,150,150", "K")
        add_shaded_trace(fig, impedance_data["ankle_K"], "Ankle K", 5, 2, "150,150,0", "K")
        add_shaded_trace(fig, impedance_data["knee_target"], "Knee Target", 6, 1, "0,150,150", "Rad")
        add_shaded_trace(fig, impedance_data["ankle_target"], "Ankle Target", 6, 2, "150,150,0", "Rad")

    fig.update_layout(height=400 * rows, width=1200, title_text="MuJoCo Biomechanics Evaluation Dashboard", showlegend=True)
    
    output_path = os.path.abspath("biomechanics_dashboard.html")
    fig.write_html(output_path)
    
    print(f"\n" + "="*80)
    print(f"BIOMECHANICS ANALYSIS COMPLETE")
    print(f"Dashboard saved to: {output_path}")
    print(f"Indices: Knee R Act={knee_act_idx} (Gear={gear_knee:.1f}), Knee L DOF={knee_l_dof}")
    print(f"="*80 + "\n")
