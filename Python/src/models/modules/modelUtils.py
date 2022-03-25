import torch
from enum import Enum
from src.models.modules.DQN import DQN, DQN_occupancy_grid
import src.utils.utils as utils
import numpy as np

class ModelVersions(Enum):
  CamPos_ObsPos    =  1
  CamPos_ObsOG     =  2
  CamPos_ObsSDF    =  3
  CamOF_ObsSDF     =  4 # opticFlow       from cam  | SDF from env
  CamOFRGB_ObsSDF  =  5 # opticFlow + RGB from cam  | SDF from env
  NoVersion        = -1
  
  @staticmethod
  def findmVersions(typeCam, typeObs):
    if   typeObs == 'Position' and typeCam == 'Position':
        return ModelVersions.CamPos_ObsPos
    elif typeObs == 'OccupancyGrid' and typeCam == 'Position':
        return ModelVersions.CamPos_ObsOG
    elif typeObs == 'SDF' and typeCam == 'Position':
        return ModelVersions.CamPos_ObsSDF
    elif typeObs == 'SDF' and typeCam == 'OF':
        return ModelVersions.CamOF_ObsSDF
    elif typeObs == 'SDF' and typeCam == 'OFRGB':
        return ModelVersions.CamOFRGB_ObsSDF
    
class StateInterpreter:

    @staticmethod
    def CamPos3_ObsPos3(mVersion, state):
      """ State interpreter: Camera Position and Cube Position in 3D
      """
      cam   = state[0][:,  :4]      # position + velocity 3 dim
      obs   = state[0][:, 4: ]      # obs 3 dim
      return obs, cam
      
    @staticmethod
    def CamPos3_ObsOG(mVersion, state):
      """ State interpreter: Camera Position and Cube Position in 3D
      """
      # import ipdb
      # ipdb.set_trace()
      
      cam   = state[0][:,  :4]      # distance + local_point 3 dim
      dims  = state[0][:, 4:7]      # dimension of occupancy grid R^3
      _obs  = state[0][:, 7: ]      # occupancy grid
      
      obs_3d = utils.reshape_occupancy_grid(_obs.cpu().numpy(),dims.cpu().numpy()[0].astype(int))
    
      
      return obs_3d, cam
    
    @staticmethod
    def CamOF_ObsSDF(mVersion, state):
      """ State interpreter: CamOF_ObsSDF
      """
      cam   = state[0][:,  :4]      # distance + local_point 3 dim
      dims  = state[0][:, 4:7]      # dimension of occupancy grid R^3
      _obs  = state[0][:, 7: ]      # occupancy grid
      
      # torch.Size([1, 480, 720, 3])
      opticFlow = state[1][:]          # 720 x 480 x 3 

      obs_3d = utils.reshape_occupancy_grid(_obs.cpu().numpy(),dims.cpu().numpy()[0].astype(int))
      
      return obs_3d, cam, opticFlow
    
    @staticmethod
    def CamOFRGB_ObsSDF(mVersion, state):
      """ State interpreter: CamOF_ObsSDF
      """
      # first observation sensor
      cam   = state[0][:,  :4]      # distance + local_point 3 dim
      dims  = state[0][:, 4:7]      # dimension of occupancy grid R^3
      _obs  = state[0][:, 7: ]      # occupancy grid
      
      # second obs sensor
      opticFlow = state[1][:]          # 720 x 480 x 3 
      
      # third obs sensor
      imgRGB    = state[2][:]          # 720 x 480 x 3 

      obs_3d = utils.reshape_occupancy_grid(_obs.cpu().numpy(),dims.cpu().numpy()[0].astype(int))
      
      return obs_3d, cam, opticFlow, imgRGB
    
class versionControl():
    """ Simple factory of state interpreters and models
    """
    
    """---------------------------------------
        class version control 
        class variable, can be called across all instances
    ---------------------------------------"""    
    current_version = ModelVersions.NoVersion
    
    def __str__(self):
      return versionControl.current_version.__str__()
      
    @classmethod
    def set_version(cls, mdVersion):
        """ seeting version for ALL class instance
        """
        versionControl.current_version = mdVersion
    
    @classmethod
    def set_version_str(cls, typeObs: str, typeCam: str):
        """ seeting version for ALL class instance by str
        """
        versionControl.current_version = ModelVersions.findmVersions(typeObs, typeCam)      
    
    """---------------------------------------
       current version access (recommanded) 
    ---------------------------------------"""    
    
    @classmethod
    def current_interpreter(cls, state):
        """ return 'current' version interpreter 
            use AFTER setting the 'current' version
        """
        assert(versionControl.current_version!=ModelVersions.NoVersion)
        return versionControl.product_interpreter(versionControl.current_version, state)
      
    @classmethod
    def current_model(cls, action_size):
        """ return 'current' version model 
            use AFTER setting the 'current' version
        """
        assert(versionControl.current_version!=ModelVersions.NoVersion)
        return cls.product_model(versionControl.current_version, action_size)
      
    """---------------------------------------
       product methods by Enum or str, 
       version specification required
       modify when adding new version
    ---------------------------------------"""  
    @staticmethod
    def product_interpreter(mdVersion, state):
        if   mdVersion == ModelVersions.CamPos_ObsPos:
            return StateInterpreter.CamPos3_ObsPos3(mdVersion, state)
        elif mdVersion == ModelVersions.CamPos_ObsOG:
            return StateInterpreter.CamPos3_ObsOG(mdVersion, state)
        elif mdVersion == ModelVersions.CamOF_ObsSDF:
            return StateInterpreter.CamOF_ObsSDF(mdVersion, state)
        elif mdVersion == ModelVersions.CamOFRGB_ObsSDF:
            return StateInterpreter.CamOFRGB_ObsSDF(mdVersion, state)
        
    @staticmethod
    def product_model(mdVersion: ModelVersions, action_size: int):
        if   mdVersion == ModelVersions.CamPos_ObsPos:
            return DQN(action_size)
        elif mdVersion == ModelVersions.CamPos_ObsOG:
            return DQN_occupancy_grid(action_size)
          
    """---------------------------------------
       apis for supporting str input etc.
    ---------------------------------------"""  
    
    @staticmethod
    def product_interpreter_str(typeObs: str, typeCam: str, state):
        return versionControl.product_interpreter(ModelVersions.findmVersions(typeObs, typeCam), state)
    @staticmethod
    def product_model_str(typeObs: str, typeCam: str, action_size: int):
        return versionControl.product_model(ModelVersions.findmVersions(typeObs, typeCam), action_size)
    