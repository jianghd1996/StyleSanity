import torch
from torch import nn
import numpy as np

class DQN(nn.Module):
    """Simple Deep Q-Network.

    :param n_actions: number of discrete actions available in the env.
    """

    def __init__(self, n_actions: int):
        super().__init__()

        self.obstacle_fc = nn.Sequential(
            nn.Linear(144, 64),
            nn.ReLU(),
        )
        self.camera_fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
        )
        self.fuse_fc = nn.Sequential(
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Linear(64, n_actions)
        )

    def forward(
        self, obstacle_state: torch.Tensor, camera_state: torch.Tensor
    ) -> torch.Tensor:
        obstacle_out = self.obstacle_fc(obstacle_state)
        camera_out   = self.camera_fc(camera_state)
        out = self.fuse_fc(torch.cat([camera_out, obstacle_out], axis=1))
        return out

class DQN_occupancy_grid(nn.Module):
  """
  Deep Q-Network with occupancy grid

  :param n_actions: number of discrete actions available in the env.
  """

  def __init__(self, n_actions: int):
    super().__init__()

    # self.dimConv3dOut=0
    self.env_encoder = nn.Sequential(
      nn.Conv3d(in_channels=1,  out_channels=8,  kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
      nn.Conv3d(in_channels=8,  out_channels=16, kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
      nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
    )
    
    self.env_fc = nn.Sequential(
      nn.Linear(32, 64),
      # nn.Linear(432, 64),
      # nn.Linear(5488, 64),
      # nn.Linear(124997, 64),
      nn.ReLU()
    )

    self.position_fc = nn.Sequential(
      nn.Linear(4, 64),
      nn.ReLU(),
    )

    self.C_value = nn.Sequential(
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, n_actions)
    )
  
  def forward(self, obstacle_state: torch.Tensor, camera_state: torch.Tensor) -> torch.Tensor:
    # print("obstacle_state,", obstacle_state.size())
    batch_size = obstacle_state.size(0)
    obstacle_state_encode = self.env_encoder(obstacle_state).view(batch_size, -1)
    # print("obstacle_state_encode,", obstacle_state_encode.size())
    
    obstacle_out = self.env_fc(obstacle_state_encode)
    
    # out single input
    # if obstacle_state_encode.size()[0]==1:
      # with open("/Users/triocrossing/INRIA/UnityProjects/DQN_PL/unity-rl-sanity-env/Python/outputs/outCube64.csv", "a") as f:
        # np.savetxt(f, obstacle_state_encode.cpu().numpy(), delimiter=",")
        # f.write(b"\n")

    camera_out = self.position_fc(camera_state)

    return self.C_value(torch.cat([obstacle_out, camera_out], axis=1))


class Discriminator(nn.Module):
    """
    Discriminator for styles

    :param n_states: maybe state + action, depends on the styles
    """

    def __init__(self, n_states : int):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=8, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x, n_x):
        x = self.conv(torch.cat([x.transpose(1, 3), n_x.transpose(1, 3)], dim=1))
        print(x.shape)

        return x

class OptClass(nn.Module):

    def __init__(self, action_space):
        super(OptClass, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(160*8, 64),
            nn.ReLU(),
            nn.Linear(64, action_space),
        )

    def forward(self, x):
        batch_size = len(x)
        x = self.conv(x.transpose(1, 3))
        x = x.view(batch_size, -1)

        return self.fc(x)