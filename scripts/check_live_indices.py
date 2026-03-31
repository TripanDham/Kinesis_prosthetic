import torch
import numpy as np
from omegaconf import OmegaConf
from src.agents.agent_gail import AgentGAIL
import mujoco

def check_live_indices():
    # Load configs and wrap them properly
    run_cfg = OmegaConf.load("/media/tripan/Data/DDP/Kinesis_prosthetic/cfg/run/myolegs_gail.yaml")
    env_cfg = OmegaConf.load("/media/tripan/Data/DDP/Kinesis_prosthetic/cfg/env/myolegs_gail.yaml")
    learn_cfg = OmegaConf.load("/media/tripan/Data/DDP/Kinesis_prosthetic/cfg/learning/gail_mlp.yaml")
    
    cfg = OmegaConf.create({
        "run": run_cfg,
        "env": env_cfg,
        "learning": learn_cfg,
        "exp_name": "test_diagnostics",
        "output_dir": "test_out",
        "no_log": True,
        "seed": 0,
        "test": True,
        "headless": True
    })
    
    # Force some params
    cfg.run.test = True
    cfg.run.device = "cpu"
    cfg.run.checkpoint = -1
    cfg.run.headless = True
    
    agent = AgentGAIL(cfg, torch.float32, "cpu", training=False)
    env = agent.env
    
    print(f"Model NQ: {env.mj_model.nq}")
    print("Joint Limits and Addresses:")
    for i in range(env.mj_model.njnt):
        name = env.mj_model.joint(i).name
        adr = env.mj_model.jnt_qposadr[i]
        limited = env.mj_model.jnt_limited[i]
        jnt_range = env.mj_model.jnt_range[i]
        print(f"Joint {i}: {name} -> qposadr {adr}, Limited: {limited}, Range: {jnt_range}")
    
    obs, info = env.reset()
    print("\nqpos values after reset():")
    for i in range(env.mj_model.njnt):
        name = env.mj_model.joint(i).name
        adr = env.mj_model.jnt_qposadr[i]
        val = env.mj_data.qpos[adr]
        print(f"{name}: {val:.4f}")
if __name__ == "__main__":
    check_live_indices()
