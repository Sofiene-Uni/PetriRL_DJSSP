\# PetriRL-DJSSP: Dynamic Job Shop Scheduling with Petri Nets and Reinforcement Learning



\*\*PetriRL-DJSSP\*\* is a novel framework for solving \*\*Dynamic Job Shop Scheduling Problems (DJSSPs)\*\* under uncertainty, where \*\*stochastic job arrivals\*\* and \*\*unexpected machine breakdowns\*\* disrupt traditional schedules.



By integrating \*\*Coloured-Timed Petri Nets (CTPNs)\*\* with \*\*Maskable Proximal Policy Optimization (MPPO)\*\*, PetriRL-DJSSP enables \*\*dynamic action masking\*\*, allowing rapid adaptation to disruptions without retraining. The result is a resilient and scalable scheduling system for real-time manufacturing environments.



---



\## Why PetriRL-DJSSP?



Dynamic scheduling requires robust and adaptive methods to handle unpredictable conditions. PetriRL-DJSSP provides:



\- \*\*Dynamic Action Masking\*\* – Prevents infeasible scheduling decisions caused by job arrivals, machine breakdowns, or unavailable resources.

\- \*\*Adaptive Scheduling\*\* – RL policies remain effective under disruptions without requiring retraining.

\- \*\*Realistic Industrial Modeling\*\*

&nbsp; - \*\*Job Arrivals:\*\* Simulated with a \*\*Gamma distribution\*\* to capture clustering and bursty patterns.

&nbsp; - \*\*Machine Failures:\*\* Simulated with a \*\*Weibull distribution\*\* to model age-related degradation and wear-out.

\- \*\*Explainability\*\* – Petri Net semantics and token flows provide interpretable insights into decision-making.

\- \*\*Scalability \& Modularity\*\* – Built on composable Petri Nets, allowing extensions to more complex systems.



---



\## Framework Overview



!\[Framework](https://github.com/Sofiene-Uni/PetriRL\_DJSSP/blob/main/framework.png)



The \*\*Petri Net\*\* serves as the backbone of the simulator:



\- \*\*Transitions\*\* → represent the RL agent’s action space.

\- \*\*Markings\*\* → encode system states and observations.

\- \*\*Guard Functions\*\* → provide dynamic action masking for feasibility.



This integration improves sample efficiency, interpretability, and adaptability to dynamic conditions.



---



\## Key Features



\- \*\*CTPN + MPPO Integration\*\*  

&nbsp; Combines interpretable Petri Net models with adaptive RL for real-time decision-making.



\- \*\*Dynamic Action Masking\*\*  

&nbsp; Extends Maskable PPO to handle disruptions caused by stochastic events, maintaining efficient scheduling.



\- \*\*Stochastic Event Modeling\*\*  

&nbsp; - Random job arrivals via \*\*Gamma distribution\*\*  

&nbsp; - Machine breakdowns via \*\*Weibull distribution\*\*



\- \*\*Benchmarking \& Reproducibility\*\*  

&nbsp; Open-source, \*\*Gym-compatible Python package\*\* available on PyPI, enabling reproducible research and fair algorithm comparisons.



\- \*\*Performance\*\*  

&nbsp; Outperforms dispatching heuristics on makespan minimization and shows higher fault tolerance under uncertainty.



---



\## Installation



```bash

pip install petrirl



