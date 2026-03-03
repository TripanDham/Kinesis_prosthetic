import mujoco
import numpy as np

xml_path = "data/xml/myolegs_OSL_KA.xml"
try:
    model = mujoco.MjModel.from_xml_path(xml_path)
    print("Joint names:")
    for i in range(model.njnt):
        print(f"{i}: {model.jnt(i).name} (qposadr: {model.jnt_qposadr[i]})")
except Exception as e:
    print(f"Error: {e}")
