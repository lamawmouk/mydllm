import os
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.utils import *
from src.runner import register_runner, get_overlay_for_save
import numpy as np


@register_runner("rtp_smc")
class RTPSMC():
    """GG-RTP-SMC: Gradient-Guided Reward-Tilted Proposal SMC.

    Combines:
    - pCNL reward-aware initialization (from Psi-Sampler, unchanged)
    - Gradient-guided M-candidate generation per particle per step
    - Softmax reward-tilted selection with proper importance weight correction

    When M=1 and eta=0, reduces exactly to the original Psi-Sampler SMC.
    """

    @ignore_kwargs
    @dataclass
    class Config():
        num_inference_steps: int = 25
        alpha: float = 0.1
        num_particles: int = 10
        ess_threshold: float = 0.5
        smc_grad_norm: float = None
        grad_const_scale: float = None
        smc_grad_const_scale: float = None

        save_tweedies: bool = False
        save_dir: str = None
        misc_dir: str = None
        return_max_reward: bool = True
        init_weight_method: str = "reward"
        grad_scale: float = 1.0

        reward_name: str = None

        # GG-RTP-SMC specific
        M: int = 4              # candidates per particle per step
        eta: float = 1.0        # gradient guidance scale for candidate generation

    def __init__(self, CFG):
        self.cfg = self.Config(**CFG)
        self.particle_schedule = [self.cfg.num_particles] * self.cfg.num_inference_steps

    def pretrained_over_proposal(self, latents, mu, sigma, grad_reward):
        """Log-ratio of pretrained transition over gradient-shifted proposal.

        This is the standard SOC correction term from Psi-Sampler:
        log p(x|mu) / q(x|mu_tilde) where mu_tilde = mu + eta*sigma^2*grad/alpha
        """
        batch_size = latents.shape[0]
        latents = latents.view(batch_size, -1)
        mu = mu.view(batch_size, -1)
        sigma = sigma.view(batch_size, -1)
        grad_reward = grad_reward.view(batch_size, -1)

        exponent = grad_reward * (-latents / self.cfg.alpha + mu / self.cfg.alpha + sigma ** 2 * grad_reward / (2 * self.cfg.alpha ** 2))
        exponent = exponent.sum(dim=-1)
        return exponent

    @torch.enable_grad()
    def run(self, pipe, time_sampler, reward_model, latents, generator=None, **kwargs):
        assert latents.shape[0] == self.particle_schedule[0], "Number of particles must match the number of latents"

        M = self.cfg.M
        eta = self.cfg.eta

        # Step 0: Initial reward/gradient evaluation (same as Psi-Sampler)
        t, dt = time_sampler(torch.tensor([0], device=latents.device, dtype=torch.int32).expand(latents.shape[0]))
        cur_reward_values, cur_grad_reward, vel_pred, decoded_tweedies = pipe.get_reward_grad_vel_tweedies(latents, reward_model, t)

        if self.cfg.save_tweedies:
            bench_idx = kwargs.get("idx", None)
            dir_name = "rtp_tweedies" if bench_idx is None else f"rtp_tweedies_{bench_idx:05d}"
            os.makedirs(os.path.join(self.cfg.misc_dir, dir_name), exist_ok=True)
            if self.cfg.reward_name == "t2icount":
                decoded_tweedies_for_save, reward_for_save = get_overlay_for_save(reward_model, decoded_tweedies)
                save_collage(decoded_tweedies_for_save, os.path.join(self.cfg.misc_dir, dir_name, f"{0:05d}.png"), resize=(256, 256), reward_values=reward_for_save)
            else:
                save_collage(decoded_tweedies, os.path.join(self.cfg.misc_dir, dir_name, f"{0:05d}.png"), resize=(256, 256), reward_values=cur_reward_values)

        # Initialize weights (same as Psi-Sampler)
        if self.cfg.init_weight_method == "original":
            cur_weights = cur_reward_values / self.cfg.alpha if pipe.cfg.mcmc == "gaussian" else torch.ones_like(cur_reward_values)
        elif self.cfg.init_weight_method == "reward":
            cur_weights = cur_reward_values / self.cfg.alpha
        else:
            raise ValueError(f"Unknown init_weight_method: {self.cfg.init_weight_method}")

        assert cur_weights.ndim == 1, "Dimension of reward values should be 1D"

        for step in tqdm(range(1, self.cfg.num_inference_steps + 1), desc="RTP-SMC Steps", leave=False):
            cur_num_particles = latents.shape[0]
            next_num_particles = self.particle_schedule[step - 1]

            # Normalize weights for ESS computation
            lse = torch.logsumexp(cur_weights - cur_weights.max(), dim=0)
            if not torch.isfinite(lse):
                lse = torch.tensor(0., device=cur_weights.device)
            normalized_weights = torch.exp(cur_weights - cur_weights.max() - lse)

            ess = 1.0 / torch.sum(normalized_weights ** 2)

            # Logging
            bench_idx = kwargs.get("idx", None)
            file_name = "values" if bench_idx is None else f"values_{bench_idx:05d}"

            if self.cfg.misc_dir is not None:
                with open(os.path.join(self.cfg.misc_dir, file_name + ".txt"), "a") as f:
                    weight_vals = [str(tmp) for tmp in cur_weights.tolist()]
                    norm_weight_vals = [str(tmp) for tmp in normalized_weights.tolist()]
                    weight_val_str = ", ".join(weight_vals)
                    norm_weight_str = ", ".join(norm_weight_vals)
                    f.write(f"Step {step}: {ess.item()}\n")
                    f.write(f"weights: {weight_val_str}\n")
                    f.write(f"Normalized Weights: {norm_weight_str}\n")
                    f.write("#" * 50 + "\n\n")

            # ESS check + SSP resampling (same as Psi-Sampler)
            if ess.item() < self.cfg.ess_threshold * cur_num_particles:
                ancestor_idx = torch.from_numpy(ssp(W=np.array(normalized_weights.cpu()), M=next_num_particles)).to(latents.device)
                prev_weights = torch.zeros_like(cur_weights)

                latents = latents[ancestor_idx]
                vel_pred = vel_pred[ancestor_idx]
                cur_reward_values = cur_reward_values[ancestor_idx]
                cur_grad_reward = cur_grad_reward[ancestor_idx]
            else:
                prev_weights = cur_weights

            N = latents.shape[0]

            if M > 1:
                # ---- GG-RTP-SMC: Multi-candidate with reward-tilted selection ----

                # Generate M gradient-guided candidates per particle
                candidates, mu, mu_tilde, sigma_dt = pipe.step_multi_candidate(
                    latents, t, dt, vel_pred, M, cur_grad_reward, eta, self.cfg.alpha
                )

                # Get next timestep for Tweedie evaluation
                if step < self.cfg.num_inference_steps:
                    t_next, dt_next = time_sampler(torch.tensor([step], device=latents.device, dtype=torch.int32).expand(N))
                else:
                    t_next = t  # last step, use current t for evaluation

                # Evaluate and select candidates via softmax reward-tilted selection
                selected, selected_rewards_from_candidates, log_q_selected, all_rewards = \
                    pipe.evaluate_and_select_candidates(candidates, reward_model, t_next if step < self.cfg.num_inference_steps else t, self.cfg.alpha, N, M)

                latents = selected

                # Now compute full reward/gradient at the selected position
                if step < self.cfg.num_inference_steps:
                    t, dt = t_next, dt_next
                    next_reward_values, next_grad_reward, vel_pred, decoded_tweedies = pipe.get_reward_grad_vel_tweedies(latents, reward_model, t)
                else:
                    next_reward_values, _, _, _ = pipe.get_reward_grad_vel_tweedies(latents, reward_model, t, return_grad=False)
                    next_grad_reward = cur_grad_reward  # won't be used

                if self.cfg.save_tweedies and step < self.cfg.num_inference_steps:
                    bench_idx = kwargs.get("idx", None)
                    dir_name = "rtp_tweedies" if bench_idx is None else f"rtp_tweedies_{bench_idx:05d}"
                    if self.cfg.reward_name == "t2icount":
                        tweedies_for_save, reward_for_save = get_overlay_for_save(reward_model, decoded_tweedies)
                        save_collage(tweedies_for_save, os.path.join(self.cfg.misc_dir, dir_name, f"{step:05d}.png"), resize=(256, 256), reward_values=reward_for_save)
                    else:
                        save_collage(decoded_tweedies, os.path.join(self.cfg.misc_dir, dir_name, f"{step:05d}.png"), resize=(256, 256), reward_values=next_reward_values)

                # ---- Importance weight correction ----

                # 1. Selection correction: log(1/M) - log(softmax(r_j/alpha))
                #    = -log(M) - log_q_selected
                selection_correction = -math.log(M) - log_q_selected  # [N]

                # 2. Gradient shift correction: log p(x|mu) / q(x|mu_tilde)
                shift_correction = self.pretrained_over_proposal(latents, mu, sigma_dt, cur_grad_reward)

                # 3. Standard SOC reward incremental term
                reward_increment = (next_reward_values - cur_reward_values) / self.cfg.alpha

                cur_weights = prev_weights + reward_increment + selection_correction + shift_correction

            else:
                # ---- M=1 fallback: exactly Psi-Sampler behavior ----
                latents, mu, sigma = pipe.step(latents, t, dt, vel_pred, return_mu_sigma=True)
                latents = latents + sigma ** 2 * cur_grad_reward / self.cfg.alpha * self.cfg.grad_scale

                if step < self.cfg.num_inference_steps:
                    t, dt = time_sampler(torch.tensor([step], device=latents.device, dtype=torch.int32).expand(latents.shape[0]))
                    next_reward_values, next_grad_reward, vel_pred, decoded_tweedies = pipe.get_reward_grad_vel_tweedies(latents, reward_model, t)
                    if self.cfg.save_tweedies:
                        bench_idx = kwargs.get("idx", None)
                        dir_name = "rtp_tweedies" if bench_idx is None else f"rtp_tweedies_{bench_idx:05d}"
                        if self.cfg.reward_name == "t2icount":
                            tweedies_for_save, reward_for_save = get_overlay_for_save(reward_model, decoded_tweedies)
                            save_collage(tweedies_for_save, os.path.join(self.cfg.misc_dir, dir_name, f"{step:05d}.png"), resize=(256, 256), reward_values=reward_for_save)
                        else:
                            save_collage(decoded_tweedies, os.path.join(self.cfg.misc_dir, dir_name, f"{step:05d}.png"), resize=(256, 256), reward_values=next_reward_values)
                else:
                    next_reward_values, _, _, _ = pipe.get_reward_grad_vel_tweedies(latents, reward_model, t, return_grad=False)
                    next_grad_reward = cur_grad_reward

                cur_weights = prev_weights + (next_reward_values - cur_reward_values) / self.cfg.alpha + self.pretrained_over_proposal(latents, mu, sigma, cur_grad_reward)

            cur_reward_values = next_reward_values.clone()
            cur_grad_reward = next_grad_reward.clone()

        max_weight_idx = torch.argmax(cur_weights).item() if not self.cfg.return_max_reward else torch.argmax(cur_reward_values).item()
        best_latents = latents[max_weight_idx: max_weight_idx + 1]
        best_sample_reward = cur_reward_values[max_weight_idx]
        return best_latents, best_sample_reward
