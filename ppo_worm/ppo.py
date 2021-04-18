import os
import yaml
import torch
import inspect
import datetime
import numpy as np
import torch.nn as nn
import mlflow.pytorch

from tqdm import tqdm
from mpi4py import MPI
from torch.optim import Adam

from utils.core import PPOBuffer
from ppo_worm.agent import PPOActorCritic
from ppo_worm.env_wrapper import WormGymWrapper
from utils.mpi_tools import (
    mpi_statistics_scalar,
    mpi_fork,
    mpi_avg,
    proc_id,
    num_procs,
    proc_id,
    setup_pytorch_for_mpi,
    sync_params,
    mpi_avg_grads
)


class PPO:
    def __init__(self, env_fn, actor_fn, model_path=None, steps_per_epoch=4000, epochs=50, gamma=0.99,
        clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97,
        max_ep_len=1000, target_kl=0.01, save_freq=10):
        self.env_fn = env_fn
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.lam = lam
        self.max_ep_len = max_ep_len
        self.target_kl = target_kl
        self.save_freq = save_freq

        setup_pytorch_for_mpi()

        self.env = self.env_fn()
        self.ac = actor_fn(self.env.observation_space, self.env.action_space)
        sync_params(self.ac)

        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=self.vf_lr)

        if model_path is None:
            # New Model
            now = datetime.datetime.now()
            now_str = now.strftime("%Y%m%d_%H:%M:%S")
            self.model_path = f"experiments/{now_str}_ppo"
        else:
            # Existing model
            self.model_path = model_path
            self.load_model(self.model_path)
        
    def compute_loss_pi(self, data):
        """Compute the loss of the actor policy.

        Args:
            data (dict): batchs of observations, actions, advantages and log probs

        Returns:
            Tuple of the policy loss and info dictionary for tensorboard logging.
        """
        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()

        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
    
        return loss_pi, pi_info

    def compute_loss_v(self, data):
        """Compute the loss of the value critic.

        Args:
            data (dict): batch of agent-environment information

        Returns:
            Value loss.
        """
        obs, ret = data["obs"], data["ret"]
        return ((self.ac.v(obs) - ret)**2).mean()

    def update(self, data):
        """Run gradient descent on the actor and critic.

        Args:
            data (dict): batch of agent-environment information

        Returns:
            Policy loss, value loss, KL-divergence, entropy and clip fraction.
        """
        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data).item()


        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = mpi_avg(pi_info["kl"])
            if kl > 1.5 * self.target_kl:
                print(f"Early stopping at {i} due to reaching max kl")
                break
            loss_pi.backward()
            mpi_avg_grads(self.ac.pi)
            self.pi_optimizer.step()

        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(self.ac.v)
            self.vf_optimizer.step()

        kl, ent, cf = pi_info["kl"], pi_info_old["ent"], pi_info["cf"]
        return pi_l_old, v_l_old, kl, ent, cf
    
    def train(self):
        """Run training across multiple environments using MPI.
        """
        comm = MPI.COMM_WORLD

        # Save parameters to YAML if the root process.
        if proc_id() == 0:
            self.log_params()

        seed = 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)
        local_steps_per_epoch = int(self.steps_per_epoch / num_procs())
        
        obs_dim = self.env.observation_space[0]
        act_dim = self.env.action_space[0]
        replay = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, self.gamma, self.lam)
        pbar = tqdm(range(self.epochs), ncols=100)
    
        # Initial observation
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        for epoch in pbar:
            episode_lengths = []
            episode_rewards = []

            for t in range(local_steps_per_epoch):
                a, v, logp = self.ac.step(torch.as_tensor(o, dtype=torch.float32))

                next_o, r, d, _ = self.env.step(a)
                ep_ret += r
                ep_len += 1

                replay.store(o, a, r, v, logp)
                o = next_o

                timeout = ep_len == self.max_ep_len
                terminal = d or timeout
                epoch_ended = t==local_steps_per_epoch-1

                if terminal or epoch_ended:
                    if epoch_ended and not(terminal):
                        print(f"Warning: trajectory cut off by epoch at {ep_len} steps.", flush=True)

                    if timeout or epoch_ended:
                        _, v, _ = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
                    else:
                        v = 0
                    replay.finish_path(v)
                    episode_lengths.append(ep_len)
                    episode_rewards.append(ep_ret)
                    o, ep_ret, ep_len = self.env.reset(), 0, 0

            
            # Gather epoch data from each process
            if proc_id() == 0:
                # Receive rewards and episode lengths
                for id in range(1, num_procs()):
                    extra_rewards = comm.recv(source=id, tag=420)
                    extra_lengths = comm.recv(source=id, tag=1337)
                    episode_rewards.extend(extra_rewards)
                    episode_lengths.extend(extra_lengths)
            else:
                # Send rewards and episode_lengths
                comm.send(episode_rewards, dest=0, tag=420)
                comm.send(episode_lengths, dest=0, tag=1337)

            pbar.set_postfix(
                dict(
                    avg_epsiode_length=f"{np.mean(episode_lengths): .2f}", 
                    avg_reward=f"{np.mean(episode_rewards): .2f}"
                )
            )

            data = replay.sample()
            pi_loss, value_loss, kl_div, entropy, clip_fraction = self.update(data)

            metrics = { 
                "Environment/Episode Length": np.mean(episode_lengths),
                "Environment/Cumulative Reward": np.mean(episode_rewards),
                "Loss/Policy": pi_loss,
                "Loss/Value": value_loss,
                "Metrics/KL Divergence": kl_div,
                "Metrics/Entropy": entropy,
                "Metrics/Clip Fraction": clip_fraction,
            }
            episode_lengths = []
            episode_rewards = []

            if proc_id() == 0:
                self.log_summary(epoch, metrics)
                if ((epoch % self.save_freq == 0) or (epoch == self.epochs - 1)):
                    self.save_model()
            
    def log_params(self):
        """Log training parameters into the YAML file for later reference.
        """
        attributes = inspect.getmembers(self, lambda a: not(inspect.isroutine(a)) and type(a) in (int, str, float))
        config = [a for a in attributes if not(a[0].startswith("__") and a[0].endswith("__"))]
        config = dict(config)
        config["algorithm"] = "ppo"
        for key, value in config.items():
            mlflow.log_param(key, value)

    def log_summary(self, epoch, metrics):
        """Log metrics onto tensorboard.
        """
        for name, value in metrics.items():
            mlflow.log_metric(name, value, epoch)
    
    def save_model(self):
        """Save model to the model directory.
        """
        mlflow.pytorch.log_model(self.ac, "model")


    def test_model(self, model_path, test_episodes):
        """Rollout the model on the environment for a fixed number of epsiodes.
        
        Args:
            model_path (str): path to the model directory
            test_episodes (int): number of episodes to run
        """
        self.env = self.env_fn()
        self.ac = self.actor_fn(self.env.observation_space, self.env.action_space)
        self.ac.load_state_dict(torch.load(f"{model_path}/actor_critic"))

        for j in range(test_episodes):
            o, d, ep_ret, ep_len = self.env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                with torch.no_grad():
                    a = self.ac.act(torch.as_tensor(o, dtype=torch.float32))
                o, r, d, _ = self.env.step(a)
                ep_ret += r
                ep_len += 1


def main():
    model_path=None
    agent_file = "worms/worms.x86_64"
    if model_path is None:
        cpus = 4
        mpi_fork(cpus)
        env_fn = lambda: WormGymWrapper(agent_file, no_graphics=True)
        ppo = PPO(env_fn, PPOActorCritic, epochs=5)
        if proc_id() == 0:
            with mlflow.start_run() as run:
                ppo.train()
        else:
            ppo.train()
    else:
        cpus = 1
        mpi_fork(cpus)
        env_fn = lambda: WormGymWrapper(agent_file, time_scale=1., no_graphics=False)
        ppo = PPO(env_fn, PPOActorCritic)
        test_episodes = 10
        ppo.test_model(model_path, test_episodes)
 
if __name__ == "__main__":
    main()


