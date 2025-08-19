import  os
import time
import petrirl 
import gymnasium as gym
from datetime import datetime
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import get_linear_fn
import torch



def train_jssp( render_mode="solution",
                benchmark="Solax",
                instance_id="sl01",
                timesteps=1e5,
                reward_f= "G",
                random_arrival= True,
                machine_down=True,
                dynamic=False,
                size=(10,10,0,0,0),
               ):
    
    env = gym.make("petrirl-dft-v0",
                   render_mode=render_mode,
                   benchmark=benchmark,
                   instance_id=instance_id,
                   reward_f=reward_f,
                   random_arrival= random_arrival,
                   machine_down=machine_down,
                   dynamic=dynamic,
                   size=size,
    
    ).unwrapped
    
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    log_file = f"logs/Log_{instance_id}_{timesteps}_{current_datetime}.zip"

    
    policy_kwargs = dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
    activation_fn=torch.nn.ReLU
    )

    
    
    # Initialize the model
    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,  # your JSSP environment
        learning_rate=3e-4,
        gamma = 1,
        gae_lambda=1,
        clip_range=1,
        ent_coef = 0.1, 
        n_steps=256,
        batch_size=2560,
        n_epochs=10,
        seed=101,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_file
    )

    start_time = time.time()  
    
    model.learn(total_timesteps=timesteps)
    end_time = time.time()  
    elapsed_time = end_time - start_time  
  
        
    # Check if the 'info' file exists
    if os.path.exists('0-info.txt'):
        with open('0-info.txt', 'a') as info_file:
            info_file.write(f" {current_datetime} -The total {instance_id} training time (seconds): {elapsed_time}\n")
    else:
        with open('0-info.txt', 'w') as info_file:
            info_file.write(f" {current_datetime} -The total {instance_id} training time (seconds): {elapsed_time}\n")
      
    print(f"Training took {elapsed_time} seconds")
    model.save(f"agents/Agent_{instance_id}_{timesteps}_{reward_f}_{int(random_arrival)}{int(machine_down)}_GAE.zip")
    


if __name__ == "__main__":
    
    
    benchmark = "Raj"
    render_mode = "solution"
    reward_f = "G"
    random_arrival = False
    machine_down = False
    dynamic = False
    size = (10,10,0,0,0)
    
    def run_training(instances, timesteps):
        for instance_id in instances:
            train_jssp(
                render_mode=render_mode,
                benchmark=benchmark,
                instance_id=instance_id,
                timesteps=timesteps,
                reward_f=reward_f,
                random_arrival=random_arrival,
                machine_down=machine_down,
                dynamic=dynamic,
                size=size,
            )
            
    # Group instances by training timesteps
    group_3M = [f"ra{str(i).zfill(2)}" for i in range(3, 21)]
    #group_3M = ["ra01"]
    
    #group_5M = ["sl00"]
    #group_10M = ["sl30", "sl40", "sl50", "sl60"]
    #group_20M = ["sl70"]   
    
    
    # Run training on each group with specified timesteps
    run_training(group_3M, timesteps=1e6)
    #run_training(group_5M, timesteps=5e6)
    #run_training(group_10M, timesteps=10e6)
    #run_training(group_20M, timesteps=20e6)
    
    



