import numpy as np
from physics_sim import PhysicsSim
import math
class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        print(self.target_pos)
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        
        
        
#        rewardvxy = -abs(self.sim.v[0]+self.sim.v[1]) #对左右速度惩罚
#        rewardtz = -0.1*(abs(self.sim.pose[2]-self.target_pos[2]))#对不在目标高度位置惩罚
#        rewardxy = -(abs(self.sim.pose[:2] - self.target_pos[:2])).sum()#对不在目标位置x，y惩罚
#        rewardang= -(abs(self.sim.angular_v[:3])).sum()*0.1
#        rewardvz = self.sim.v[2]*2     #给向上速度奖励。
        
#        all_reward = rewardvz+rewardvxy+rewardtz+rewardxy+rewardang
#        rewardvxy = -abs(self.sim.v[0]+self.sim.v[1])
#        rewardvz = self.sim.v[2]
#        rewardxy = -(abs(self.sim.pose[:2] - self.target_pos[:2])).sum()#对不在目标位置x，y惩罚
        

        """Uses current pose of sim to return reward."""
       
        # Reward positive velocity along z-axis
        
        reward_vz = self.sim.v[2]*1.3-self.sim.v[1]*0.2-self.sim.v[0]*0.2
        
        
        # Reward positions close to target along z-axis
        
        reward_z = -(abs(self.sim.pose[2] - self.target_pos[2]))
        
        
        # Reward positions close to target in xy-plane
        reward_x = -(abs(self.sim.pose[0] - self.target_pos[0]))*0.8
        
        reward_y = -(abs(self.sim.pose[1] - self.target_pos[1]))*0.8
        
        
        # Reward ang change smoothly
        #reward_ang= -(abs(self.sim.angular_v[:3])).sum()*0.2 
        reward_ang= -(abs(self.sim.angular_v[:3])).sum()*0.8
        
        # Reward additional -1 to punish for each step, to reach target fast
        return reward_vz+reward_z+reward_x+reward_y+reward_ang-1.0


        

    def step(self, rotor_speeds):
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
            
            # 设置向上飞的奖励
            if(self.sim.pose[2] >= self.target_pos[2]):
                reward +=100
                done = True           
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def getPose(self):
        
        return self.sim.pose
    


    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state