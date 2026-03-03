# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.

import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
from src.agents.agent_humanoid import AgentHumanoid
from src.env.myolegs_IL import MyoLegsGAIL
from src.KinesisCore.expert_dataset import get_expert_loader
from src.learning.learning_utils import to_train, to_test, to_device

logger = logging.getLogger(__name__)

class AgentGAIL(AgentHumanoid):
    """
    AgentGAIL integrates GAIL discriminator training into the AgentHumanoid framework.
    """
    
    def __init__(self, cfg, dtype, device, training: bool = True, checkpoint_epoch: int = 0):
        super().__init__(cfg, dtype, device, training, checkpoint_epoch)
        
        # Expert buffer for discriminator training
        if training:
            self.history_len = cfg.env.get("history_len", 3)
            self.batch_size_disc = cfg.learning.get("batch_size_disc", 64)
            self.loader_exp = get_expert_loader(
                path=cfg.run.expert_buffer_path,
                batch_size=self.batch_size_disc,
                history_len=self.history_len,
                shuffle=True
            )
            self.iter_exp = iter(self.loader_exp)
            self.epoch_disc = cfg.learning.get("epoch_disc", 10)
    def setup_env(self):
        """
        Initializes the MyoLegsGAIL environment based on the configuration.
        """
        self.env = MyoLegsGAIL(self.cfg)
        logger.info("MyoLegsGAIL environment initialized.")

    def update_params(self, batch) -> float:
        """
        Extends parameter updates with GAIL discriminator training.
        """
        t0 = time.time()
        
        # 1. Update Discriminator
        if self.training:
            self.train_discriminator(batch)
        
        # 2. Update Policy and Value (Standard PPO)
        # Note: In AgentIM/PPO, update_params handles the conversion to tensors and calls update_policy.
        super().update_params(batch)
        
        return time.time() - t0

    def train_discriminator(self, batch):
        """
        Trains the discriminator using agent rollouts and expert demonstrations.
        """
        to_train(self.env.gail_disc)
        
        # We need to extract (s, s') or just s from the batch for the discriminator.
        # However, our environment's GAILDiscrim expects concatenated history observations.
        # Agent observations in the batch might not be the history-concatenated ones 
        # depending on how compute_task_obs is implemented.
        
        # In our MyoLegsIm.compute_reward, we maintain self.history_buffer.
        # But the batch contains what was returned by step().
        
        for _ in range(self.epoch_disc):
            # Sample from agent's batch
            # Assuming batch.states contains the history-concatenated observations
            # if that's what compute_task_obs returns.
            # However, compute_task_obs returns a concatenated vector of length obs_size.
            # The GAIL rewards in compute_reward use self.get_obs() which is often full state.
            
            # Let's assume the agent batch contains the same "state" used by the discriminator.
            indices = np.random.choice(len(batch.states), self.batch_size_disc, replace=False)
            states_pi = torch.from_numpy(batch.states[indices]).to(self.dtype).to(self.device)
            # State-only GAIL uses a dummy action
            actions_pi = torch.zeros((self.batch_size_disc, 0), device=self.device)
            
            # Sample from expert buffer
            try:
                states_exp = next(self.iter_exp)
            except StopIteration:
                self.iter_exp = iter(self.loader_exp)
                states_exp = next(self.iter_exp)
            
            states_exp = states_exp.to(self.dtype).to(self.device)
            # GAILDiscrim.forward(states, actions)
            
            # Update discriminator
            logits_pi = self.env.gail_disc(states_pi, actions_pi)
            logits_exp = self.env.gail_disc(states_exp, actions_pi) # Using dummy actions for both
            
            loss_pi = -F.logsigmoid(-logits_pi).mean()
            loss_exp = -F.logsigmoid(logits_exp).mean()
            loss_disc = loss_pi + loss_exp
            
            self.env.optim_disc.zero_grad()
            loss_disc.backward()
            self.env.optim_disc.step()
            
        logger.debug(f"Discriminator loss: {loss_disc.item():.4f}")
