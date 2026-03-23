import mujoco
import numpy as np
import os
from PIL import Image

def capture_ankle_comparison(xml_path, output_dir):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Joint Names
    human_ankle = "ankle_angle_l"
    osl_ankle = "osl_ankle_angle_r"
    
    # Get Joint IDs
    human_id = mujoco.mj_name2id(model, mujoco.mju_str2Type("joint"), human_ankle)
    osl_id = mujoco.mj_name2id(model, mujoco.mju_str2Type("joint"), osl_ankle)
    
    # Print Info
    print(f"Human Ankle ID: {human_id}, Range: {model.jnt_range[human_id]}, Axis: {model.jnt_axis[human_id]}")
    print(f"OSL Ankle ID: {osl_id}, Range: {model.jnt_range[osl_id]}, Axis: {model.jnt_axis[osl_id]}")
    
    # Camera setup (looking at the feet)
    scene_option = mujoco.MjvOption()
    camera = mujoco.MjvCamera()
    camera.lookat = np.array([0, 0, 0.2])
    camera.distance = 1.0
    camera.azimuth = 90
    camera.elevation = -10
    
    test_cases = [
        ("Neutral", 0.0, 0.0),
        ("Max_Dorsiflexion", 0.5236, 0.5236),
        ("Max_Plantarflexion", -0.6981, -0.5236)
    ]
    
    for label, human_val, osl_val in test_cases:
        mujoco.mj_resetData(model, data)
        
        # Set Joint Values
        data.joint(human_ankle).qpos[0] = human_val
        data.joint(osl_ankle).qpos[0] = osl_val
        
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
        
        img = Image.fromarray(pixels)
        img.save(os.path.join(output_dir, f"{label}.png"))
        print(f"Saved {label}.png")

if __name__ == "__main__":
    xml_path = "/media/tripan/Data/DDP/Kinesis_prosthetic/data/xml/myolegs_OSL_KA.xml"
    output_dir = "/media/tripan/Data/DDP/Kinesis_prosthetic/data/eval_plots/ankle_comparison"
    capture_ankle_comparison(xml_path, output_dir)
