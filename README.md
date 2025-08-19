## PetriRL-DJSSP (`petrirl-djssp-v0`)  

The **PetriRL-DJSSP** environment focuses on **Dynamic Job Shop Scheduling Problems (DJSSPs)** under uncertainty, including:  

- **Dynamic Action Masking** – Prevents infeasible scheduling decisions caused by stochastic job arrivals, machine breakdowns, or unavailable resources.  
- **Adaptive Scheduling** – RL policies remain effective under disruptions without retraining.  
- **Realistic Industrial Modeling** – Job arrivals via **Gamma distribution** and machine failures via **Weibull distribution**.  
- **Explainable Petri Net Semantics** – Token flows provide interpretable insights into decision-making.  

### Key Features  

- **Colored-Timed Petri Nets (CTPNs) + MPPO:**  
  Combines interpretable Petri Net models with Maskable Proximal Policy Optimization for real-time adaptive scheduling.  

- **Dynamic Action Masking:**  
  Extends Maskable PPO to handle stochastic events, maintaining efficient scheduling under disruptions.  

- **Stochastic Event Modeling:**  
  - Random job arrivals via **Gamma distribution**  
  - Machine breakdowns via **Weibull distribution**  

- **Benchmarking & Reproducibility:**  
  Open-source, **Gym-compatible Python package** available on PyPI for reproducible research and fair algorithm comparisons.  

- **Performance:**  
  Outperforms classical dispatching heuristics on makespan and shows higher fault tolerance under uncertainty.  

---

## Framework Overview  

![Framework](https://github.com/Sofiene-Uni/PetriRL_DJSSP/blob/main/framework.pdf)  

---

## Installation  

```bash
pip install petrirl
