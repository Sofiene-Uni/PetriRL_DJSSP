import petrirl
import pandas as pd
from tqdm import tqdm
import gymnasium as gym
import numpy as np
from scipy.stats import t
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from petrirl.utils.time_gen import save_seeds,load_seeds


def algorithm_sample(env, instance, agent_id, n_samples,timesteps,seeds_file):
    """Runs the scheduling algorithm multiple times on the same environment."""
    
    prefix = instance[:2]        
    digit = instance[2] if len(instance) > 2 else '0'  
    group_agent = f"{prefix}{digit}0"  
    agent = f"agents/Agent_{group_agent}_{timesteps}_{env.reward_f}_{int(random_arrival)}{int(machine_down)}.zip"
    

    job_makespan_samples = []  
    seeds = load_seeds (seeds_file)
    
    if len(seeds) >= n_samples:
        env.sim.set_seeds(seeds)  # set saved seeds
    else:
        raise ValueError("Not enough seeds available. Please choose a seeds file with at least "
                     f"{n_samples} seeds, but only {len(seeds)} found.")
        
    for sample_idx in tqdm(range(n_samples), desc="Taking samples"):
        
        i, terminated, (obs, info) = 0, False, env.reset()
        model = MaskablePPO.load(agent)
        
        while not terminated:
            env_masks = get_action_masks(env)
            action, _states = model.predict(obs, action_masks=env_masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            i += 1

        job_makespan_samples.append(env.sim.clock)

    return job_makespan_samples


def confidence_interval(data, confidence=0.95):
    """Compute confidence interval and variance."""
    n = len(data)
    if n == 1:
        return data[0], data[0], 0  # Avoid division by zero if only one sample
    
    mean_val = np.mean(data)
    variance = np.var(data, ddof=1)  # Sample variance (unbiased estimator)
    std_err = np.sqrt(variance) / np.sqrt(n)  # Standard error
    margin = t.ppf((1 + confidence) / 2, n - 1) * std_err  # Critical value * standard error
    
    return mean_val - margin, mean_val + margin, variance


def generate_results(benchmark, layouts, instances,reward_f, n_samples, timesteps_dict, seeds_file, results_filename, random_arrival, machine_down):
    """
    Runs the scheduling algorithm on different instances & job heuristics, averaging results over `n_samples` runs.
    Saves results progressively to prevent loss of data.
    """

    # Initialize lists to collect rows first (much faster and cleaner)
    results_rows = []
    detail_rows = []
    run_data_rows = []
    

    for instance in tqdm(instances, desc="Instances"):
        print(f"Processing instance: {instance}")
        
        timesteps=timesteps_dict[instance]
        
        

        for layout in layouts:
            row = {"Instance": instance, "Size": "-"}

            env = gym.make(
                "petrirl-dft-v0", benchmark=benchmark,
                instance_id=instance,
                reward_f=reward_f,
                render_mode="solution",
                random_arrival=random_arrival,
                machine_down=machine_down,
                
            ).unwrapped

            makespan_samples = algorithm_sample(env, instance, instance, n_samples, timesteps, seeds_file)
            cleaned_samples = [sample for sample in makespan_samples if sample != 0]

            # Compute statistics on cleaned samples
            row["PPO"] = np.mean(cleaned_samples) if cleaned_samples else 0
            min_val = np.min(cleaned_samples) if cleaned_samples else 0
            max_val = np.max(cleaned_samples) if cleaned_samples else 0
            mean_val = np.mean(cleaned_samples) if cleaned_samples else 0
            ci_low, ci_high, variance = confidence_interval(cleaned_samples) if cleaned_samples else (0, 0, 0)

            # Store main results
            results_rows.append(row)

            # Store detailed statistics
            detail_row = {
                "Instance": instance,
                "Min": min_val,
                "Max": max_val,
                "Mean": mean_val,
                "Variance": variance,
                "CI_Low": ci_low,
                "CI_High": ci_high,
                "Cleaned_Length": len(cleaned_samples)
            }
            detail_rows.append(detail_row)

            # Store run-wise makespan data
            for run, makespan in enumerate(cleaned_samples):
                run_data_rows.append({
                    "Instance": instance,
                    "Run": run + 1,
                    "Makespan": makespan
                })

        # After each instance, save results
        results_df = pd.DataFrame(results_rows, columns=["Instance", "Size", "PPO"])
        detail_df = pd.DataFrame(detail_rows, columns=["Instance", "Min", "Max", "Mean", "Variance", "CI_Low", "CI_High", "Cleaned_Length"])
        run_data_df = pd.DataFrame(run_data_rows, columns=["Instance", "Run", "Makespan"])

        with pd.ExcelWriter(results_filename, engine="xlsxwriter") as writer:
            results_df.to_excel(writer, sheet_name="Main", index=False)
            detail_df.to_excel(writer, sheet_name="Detail", index=False)
            run_data_df.to_excel(writer, sheet_name="RunData", index=False)
        print(f"Results saved after instance {instance} to {results_filename}_All")



if __name__ == "__main__":
    
    layouts = [1]  
    
    benchmark = "Solax"
    instances = ["sl00","sl10","sl20","sl30","sl40","sl50","sl60","sl70"]
    #instances = [f"sl{i:02}" for i in range(80)]

    # Training time dictionary using pattern keys
    timesteps_groups = {
        "sl0X": 5e6,
        "sl1X": 5e6,
        "sl2X": 5e6,
        "sl3X": 10e6,
        "sl4X": 10e6,
        "sl5X": 10e6,
        "sl6X": 10e6,
        "sl7X": 20e6
    }
    
    # Create final mapping
    timesteps_dict = {}
    
    for inst in instances:
        idx = int(inst[2:])           
        group = f"sl{idx // 10}X"     
        timesteps_dict[inst] = timesteps_groups[group]
        
    # benchmark = "Raj"
    # instances = [f"ra{str(i).zfill(2)}" for i in range(1,21)]
    # timesteps_dict = {f"ra{str(i).zfill(2)}": 3e6 for i in range(1, 21)}
 
    n_samples = 100
    #save_seeds(n_samples)
    random_arrival= False
    machine_down= True
    seeds_file="3e4d1000"
    reward_f="M"
    results_filename = f"R_{benchmark}_{seeds_file}_{n_samples}_{int(machine_down)}{int(random_arrival)}_{reward_f}.xlsx"
    
    # Generate results
    generate_results(benchmark=benchmark,
                      layouts=layouts,
                      instances=instances,
                      reward_f=reward_f,
                      n_samples=n_samples,
                      timesteps_dict=timesteps_dict,
                      seeds_file=seeds_file,
                      results_filename=results_filename,
                      random_arrival= random_arrival,
                      machine_down=machine_down,
                     
                      )
