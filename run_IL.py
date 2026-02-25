import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import logging
from src.agents.agent_gail import AgentGAIL

logger = logging.getLogger(__name__)

@hydra.main(config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    # Setup Device
    device = torch.device(cfg.run.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32
    
    logger.info(f"Using device: {device}")
    
    # Initialize Agent
    agent = AgentGAIL(
        cfg=cfg,
        dtype=dtype,
        device=device,
        training=not cfg.run.test,
        checkpoint_epoch=cfg.run.get("checkpoint", 0)
    )
    
    # Run
    if cfg.run.test:
        logger.info("Starting Evaluation Pipeline")
        agent.eval_policy(runs=cfg.run.get("num_eval_runs", 10))
    else:
        logger.info("Starting Training Pipeline")
        agent.optimize_policy()

if __name__ == "__main__":
    main()
