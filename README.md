# Python/TensorFlow implementations of the parallel control methods presented in the paper

Simo Särkkä and Ángel F. García-Fernández (2023). Temporal Parallelisation of Dynamic Programming and Linear Quadratic Control. IEEE Transactions on Automatic Control. Volume 68, Issue 2, 851-866. Open access: https://ieeexplore-ieee-org.libproxy.aalto.fi/document/9697418 (https://arxiv.org/abs/2104.03186)

```
@article{Sarkka_et_al_2023,
  author={S\"arkk\"a, Simo and Garc\'ia-Fern\'andez, \'Angel F.},
  journal={IEEE Transactions on Automatic Control}, 
  title={Temporal Parallelization of Dynamic Programming and Linear Quadratic Control}, 
  year={2023},
  volume={68},
  number={2},
  pages={851-866},
  doi={10.1109/TAC.2022.3147017}
}
```

The aim is also include the codes for https://arxiv.org/abs/2212.11744 here soon, but you need to wait until the paper has been published.

# Instructions

Clone the repository:

```
% git clone git@github.com:EEA-sensors/parallel-control-codes.git
```

Make conda environment and install the package:

```
% conda create --name parcon python=3.9
% conda activate parcon
% cd parallel-control-codes
% pip install .
```

Connect to jupyter-lab is you feel like it:

```
% conda install jupyterlab
% conda install -c anaconda ipykernel
% python -m ipykernel install --user --name=parcon
% jupyter-lab notebooks/ &
```

## Notebooks

Directly notebooks/ contains examples that you can run in jupyterlab:

```
finite_experiment.ipynb
linear_experiment.ipynb
lqr_iter_demo.ipynb
mass_experiment.ipynb
nonlinear_experiment.ipynb
```

Directory experiments-bellman/ contains scripts for running speed experiments. They would work just like:

```
% mkdir -p res
% cd experiments-bellman
% cpu_tracking_1.sh
```

