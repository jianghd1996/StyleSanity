from collections import OrderedDict
from typing import List, Tuple

from mlagents_envs.environment import UnityEnvironment
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import DistributedType
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from src.models.modules.DQN import Discriminator, OptClass
from src.models.memory import ReplayBuffer, RLDataset
from src.models.agent import Agent
import src.utils.utils as utils
from src.models.modules.modelUtils import versionControl as vc
import time
import math
import numpy as np
import wandb
 
# batch = states, actions, rewards, dones, next_states
BatchTuple = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


class DQNModule(LightningModule):
    """Basic DQN Model.

    :param env: Unity environment.
    :param behavior_name: name of the unity ml-agent behavior.
    :param n_actions: size of the action space.
    :param run_type: train or infer.
    :param batch_size: size of the batches.
    :param replay_size: capacity of the replay buffer.
    :param max_episodes: maximum number of episodes to run.
    :param episode_length: max length of an episode.
    :param eps_last_frame: what frame should epsilon stop decaying.
    :param eps_start: starting value of epsilon.
    :param eps_end: final value of epsilon.
    :param sync_rate: number of frames before updating the target network.
    :param lr: learning rate.
    :param lr_reduce_rate: learning rate reduction rate.
    :param gamma: discount factor.
    :param weight_decay: weight decay value.
    """

    def __init__(
        self,
        env: UnityEnvironment,
        behavior_name: str,
        n_actions: int,
        run_type: str,
        batch_size: int,
        replay_size: int,
        max_episodes: int,
        episode_length: int,
        sync_rate: int,
        eps_last_frame: int,
        eps_start: float,
        eps_end: float,
        lr: float,
        lr_reduce_rate: float,
        gamma: float,
        weight_decay: float,
    ):
        super().__init__()

        self.env = env

        # self.net = \
        #   vc.current_model(n_actions)
        # self.target_net = \
        #   vc.current_model(n_actions)

        self.net = OptClass(5)

        self.D = Discriminator(4)

        self.criterion = nn.MSELoss()
        self.cri_cate = nn.CrossEntropyLoss()

        self._lr = lr
        self._lr_reduce_rate = lr_reduce_rate
        self._weight_decay = weight_decay
        self._gamma = gamma
        self._sync_rate = sync_rate
        self._eps_last_frame = eps_last_frame
        self._eps_start = eps_start
        self._eps_end = eps_end

        self._batch_size = batch_size
        self.episode_length = episode_length
        self.max_episodes = max_episodes

        self.step_index = 0
        self.episode_index = 0
        self.total_reward = 0
        self.cumreward = 0
        self.episode_losses = []
        self.episode_D_losses = []
        self.episode_q_values = []
        self.episode_success = []
        
        self._replay_size = replay_size
        self.buffer = ReplayBuffer(replay_size)
        self.gt_buffer = ReplayBuffer(replay_size)
        self.agent = Agent(self.env, self.buffer, self.gt_buffer, n_actions, behavior_name)
        if run_type == "train":
            self.populate(self._replay_size)

    def populate(self, steps: int) -> None:
        """
        Carries out several random steps through the environment to initially
        fill up the replay buffer with experiences (on CPU).

        :param steps: number of random steps to populate the buffer with.
        """
        for i in tqdm(range(steps)):
            reward, done, _state = self.agent.play_step(self.net, epsilon=1.0, device="cpu")

            if done:
                self.agent.reset()
        
        with tqdm(total=steps) as pbar:
          while(len(self.agent.gt_buffer)<steps):
            reward, done, _state = self.agent.play_step(gt=True)
            if done:
                self.agent.reset()
            pbar.update(len(self.agent.gt_buffer) - pbar.n)

    def forward(
        self, obstacle_state: torch.Tensor, camera_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Passes in a state x through the network and gets the q_values of each
        action as an output.

        :param obstacle_state: obstacle state.
        :param camera_state: camera state.
        :return: q values.
        """
        output = self.net(obstacle_state, camera_state)
        return output

    def cate_loss(self, batch: BatchTuple) -> torch.Tensor:
        states, actions, rewards, dones, next_states = batch

        opt = states[1]

        out = self.net(opt)

        loss = self.cri_cate(out, actions)

        return loss

    def dqn_mse_loss(self, batch: BatchTuple) -> torch.Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        :param batch: current mini batch of replay data.
        :return: loss and batch averaged Q-value.
        """
        states, actions, rewards, dones, next_states = batch
        obs, cam = vc.current_interpreter(states)
        
        state_action_values = (
            self.net(obs, cam)
            .gather(1, actions.unsqueeze(-1))
            .squeeze(-1)
        )

        with torch.no_grad():
            next_obs, next_cam = vc.current_interpreter(next_states)
            next_state_values  = self.target_net(next_obs, next_cam).max(1)[0]
            
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

            n_opt = next_states[1]
            expected_state_action_values = (
                next_state_values * self._gamma + rewards + self.D(opt, n_opt).squeeze(1)
            )

        loss = self.criterion(
            state_action_values, expected_state_action_values
        )

        return loss, state_action_values.mean()

    def D_loss(self, batch: BatchTuple, r_batch: BatchTuple) -> torch.Tensor:
        states, actions, rewards, dones, next_states = batch
        r_states, r_actions, r_rewards, r_dones, r_next_states = r_batch

        # state = list [(bs x 1 x 132), (bs x 720 x 480 x 3), (bs x 720 x 480 x 3)]
        opt, n_opt = states[1], next_states[1]
        r_opt, r_n_opt = r_states[1], r_next_states[1]

        device = self.get_device(batch)
        
        # [TODO]: OF -> D
        # batch size
        valid = torch.ones(len(r_states[0]), 1).to(device)
        fake = torch.zeros(len(states[0]), 1).to(device)

        D_loss = 0.5 * (self.criterion(self.D(opt, n_opt), fake).mean()
                        +self.criterion(self.D(r_opt, r_n_opt), valid).mean())

        return D_loss


    def training_step(self, batch: BatchTuple, nb_batch: int) -> OrderedDict:
        """
        Carries out a single step through the environment to update the replay
        buffer. Then calculates loss based on the minibatch recieved.

        :param batch: current mini batch of replay data.
        :param nb_batch: batch number.
        :return: training loss and log metrics.
        """
        batch, r_batch = batch

        device = self.get_device(batch[0])

        reward, done, _state = self.agent.play_step(self.net, 1.0, device)

        if done:
            self.agent.reset()

        loss = self.cate_loss(batch).unsqueeze(0)

        self.step_index += 1

        logs = {
            "step/loss": loss.detach(),
        }

        outputs = OrderedDict({"loss": loss, "logs": logs})
        return outputs

        device = self.get_device(batch[0])

        epsilon = 0.1
        # Step through environment with agent
        reward, done, _state = self.agent.play_step(self.net, epsilon, device)
        self.cumreward += reward

        # Calculates training loss
        loss, q_value = self.dqn_mse_loss(batch)
        if self._distrib_type in {
            DistributedType.DP,
            DistributedType.DDP2,
        }:
            loss = loss.unsqueeze(0)
            q_value = q_value.unsqueeze(0)

        self.episode_losses.append(loss.detach())
        self.episode_q_values.append(q_value.detach())

        # Soft update of target network
        if self.global_step % self._sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        if self.step_index == self.episode_length:
            done = True

        self.step_index += 1

        logs = {
            "step/reward": reward,
            "step/loss": self.episode_losses[-1],
            "step/q_value": self.episode_q_values[-1],
        }

        if done:
            self.episode_success.append(int(self.step_index == self.episode_length))
            logs.update({
                "episode/cumreward": torch.tensor(self.cumreward).to(device),
                "episode/loss": torch.tensor(self.episode_losses).to(device),
                "episode/D_loss": torch.tensor(self.episode_D_losses).to(device),
                "episode/q_value": torch.tensor(self.episode_q_values).to(device),
                "episode": torch.tensor(self.episode_index).to(device),
                "episode/episode_success": self.episode_success[-1],
                "episode/epsilon": epsilon,
                "episode/steps": self.step_index,
                "episode/end_reward": reward})
            self.cumreward = 0
            self.step_index = 0
            self.episode_D_losses = []
            self.episode_losses = []
            self.episode_index += 1
            self.agent.reset()

            for i in tqdm(range(50)):
                reward, done, _state = self.agent.play_step(gt=True)

                if done:
                    self.agent.reset()

        outputs = OrderedDict({"loss": loss, "logs": logs})
        return outputs

        # else:
        #     D_loss = self.D_loss(batch, r_batch).unsqueeze(0)
        #
        #     self.episode_D_losses.append(D_loss.detach())
        #
        #     logs = {
        #         "step/D_loss": self.episode_D_losses[-1],
        #     }
        #
        #     outputs = OrderedDict({"loss": D_loss, "logs": logs})
        #     return outputs


    def training_epoch_end(self, outputs):
        """Log step or episode metrics."""
        for out in outputs:
            logs = out["logs"]
            for log_name, log_value in logs.items():
                self.log(
                    log_name,
                    log_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )
        # for out in outputs:
        #     for log in out:
        #         logs = log["logs"]
        #         for log_name, log_value in logs.items():
        #             self.log(
        #                 log_name,
        #                 log_value,
        #                 on_step=False,
        #                 on_epoch=True,
        #                 prog_bar=False,
        #                 logger=True,
        #                 sync_dist=True,
        #            )

    def test_step(self, batch: BatchTuple, nb_batch: int):
        """Infer the model."""

        device = self.get_device(batch)
        epsilon = 0
        self.episode_success = []
        # Step through environment with agent
        reward, done, _state = self.agent.play_step(self.net, epsilon, device)
        self.cumreward += reward
        self.step_index += 1
        # loss, q_value = self.dqn_mse_loss(batch)

        if self.step_index == self.episode_length:
            done = True

        if done:
            self.episode_success.append(int(self.step_index == self.episode_length))
            print(
                f"[Episode {self.episode_index + 1}/{self.max_episodes}] ",
                f"Cumreward: {self.cumreward}",
            )
            
            logs= {
                    "episode/cumreward": torch.tensor(self.cumreward).to(device),
                    # "episode/q_value": torch.tensor(self.episode_q_values).to(device),
                    "episode": torch.tensor(self.episode_index).to(device),
                    "episode/episode_success": self.episode_success[-1],
                  }            
            self.cumreward = 0
            self.step_index = 0
            # self.episode_losses = []
            self.episode_index += 1
            self.agent.reset()
        else:
            logs = {
                "step/reward": torch.tensor(reward).to(device),
                # "step/loss": self.episode_losses[-1],
                # "step/q_value": self.episode_q_values[-1],
                # "obs/angle":  utils.angle_between(_state[:3], _state[3:]), 
            }
            
        wandb.log(logs)
        # self.logger.experiment.log_metrics()

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(
            self.net.parameters(),
            lr=self._lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self._weight_decay,
            amsgrad=False,
        )
        # optimizer_D = Adam(
        #     self.D.parameters(),
        #     lr=self._lr,
        #     betas=(0.9, 0.999),
        #     eps=1e-8,
        #     weight_decay=self._weight_decay,
        #     amsgrad=False,
        # )
        scheduler = {"scheduler" : torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self._lr_reduce_rate)}
        # scheduler_D = {"scheduler" : torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer_D, gamma=self._lr_reduce_rate)}
        # return [optimizer, optimizer_D], [scheduler, scheduler_D]
        return [optimizer], [scheduler]

    def __dataloader(self, buffer) -> DataLoader:
        """
        Initialize the Replay Buffer dataset used for retrieving experiences.
        """
        dataset = RLDataset(buffer, self._replay_size)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
        )
        return dataloader

    def train_dataloader(self) -> List[DataLoader]:
        """Get train loader."""
        return [self.__dataloader(self.buffer), self.__dataloader(self.gt_buffer)]

    def test_dataloader(self) -> DataLoader:
        """Generate an "empty" dataloader to start the test."""
        return DataLoader(torch.ones(int(1e5), dtype=bool))

    def get_device(self, batch: BatchTuple) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"
      