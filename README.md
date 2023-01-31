# EscalKMN

EscalKMN stands for "Effective Energy Rescaling Kinetic Markov Network". It is a package for the analysis of extensive molecular dynamics simulations and plotting results based on Jupyter Notebook tools.

The EscalKMN algorithm mapped simulation trajectories into an orthogonal function space, whose bases were rescaled by effective energy, and clustered the interrelation between these trajectories to locate metastable states. Here the effective energy, which was filtered from the total potential energy of simulation trajectories by fast Fourier transform (FFT) and multiple linear regression, is an efficacious order parameter to describe the slow conformational change of complex system. Finally, Markov kinetic network is constructed based on the transitions between these metastable states.

## Main modules

- lib: Basic libraries for performing the EscalKMN algorithms
- notebook: Jupyter Notebook scripts for analyzing MD trajectories and plot result
- gmxtools: Scripts for executing Gromacs analysis tools in batch mode

## Dependencies

- Python, Jupyter Notebook
- Numpy, Pandas, Matplotlib, Seaborn
- Sklearn, Sklearn_extra, pyEMMA

## Citation

If you use PyEMMA in scientific work, please cite:

- Zhenyu Wang, Xin Zhou and Guanghong Zuo, “EspcTM: Kinetic Transition Network based on Trajectory Mapping in Effective Energy Rescaling Space” Frontiers in Molecular Biosciences, 7, 589718, 2020
