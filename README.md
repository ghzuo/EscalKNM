# EscalKNM

EscalKNM stands for "Effective Energy Rescaling Kinetic Network Model". It is a Python package for analyzing molecular dynamics simulations based on the effective energy and plotting results based on Jupyter Notebook tools.

The effective energy, which was filtered from the total potential energy of simulation trajectories by fast Fourier transform (FFT) and multiple linear regression, is an efficacious order parameter to describe the slow conformational change of the complex system. Recently, the EspcTM algorithm had been published. It mapped simulation trajectories into an orthogonal function space, whose bases were rescaled by effective energy, and clustered the interrelation between these trajectories to locate metastable states, and Markov kinetic network is constructed based on the transitions between these metastable states. And other analysis algorithms and tools are in development.

## Main modules

- toolkits: Basic tools for performing the EscalKNM algorithms
- notebook: Jupyter Notebook scripts  
  - EscalTM: Jupyter Notebook scripts for EscalTM method
  - EscalMSM: Jupyter Notebook scripts for EscalMSM method
  - Tools: Jupyter Notebook scripts for basic function
- gmxtools: Scripts for executing Gromacs analysis tools in batch mode

## Dependencies

- Python, Jupyter Notebook
- Numpy, Pandas, Matplotlib, Seaborn
- Sklearn, Sklearn_extra, pyEMMA

## Citation

If you use EscalKNM in scientific work, please cite:

- Zhenyu Wang, Xin Zhou and Guanghong Zuo, “EspcTM: Kinetic Transition Network based on Trajectory Mapping in Effective Energy Rescaling Space” Frontiers in Molecular Biosciences, 7, 589718, 2020
