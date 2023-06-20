# Python/TensorFlow implementations of the parallel control and Viterbi methods

The methods have been published in the following papers. If you use the codes in your experiments, feel free to cite all of them:

*[1] Simo Särkkä and Ángel F. García-Fernández (2023). Temporal Parallelisation of Dynamic Programming and Linear Quadratic Control. IEEE Transactions on Automatic Control. Volume 68, Issue 2, 851-866.*

- Open access: https://ieeexplore.ieee.org/document/9697418
- arXiv: https://arxiv.org/abs/2104.03186

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

*[2] Simo Särkkä and Ángel F. García-Fernández (2023). On The Temporal Parallelisation of The Viterbi Algorithm. To appear in proceedings of EUSIPCO.*

- Preprint: https://users.aalto.fi/~ssarkka/pub/eusipco_2023_viterbi.pdf

```
@inproceedings{Sarkka_et_al_2023b,
  author={S\"arkk\"a, Simo and Garc\'ia-Fern\'andez, \'Angel F.},
  booktitle={Proceedings of EUSIPCO},
  title={On The Temporal Parallelisation of The Viterbi Algorithm},
  year={2023},
}
```

*[3] Simo Särkkä, Ángel F. García-Fernández. Temporal Parallelisation of the HJB Equation and Continuous-Time Linear Quadratic Control. arXiv:2212.11744*

```
@misc{Sarkka_et_al_202x,
  author={S\"arkk\"a, Simo and Garc\'ia-Fern\'andez, \'Angel F.},
  note={arXiv:2212.11744},
  title={Temporal Parallelisation of the HJB Equation and Continuous-Time Linear Quadratic Control},
}
```

- arXiv: https://arxiv.org/abs/2212.11744


# Instructions for installing

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

# Notebooks

Directory notebooks/ contains messy examples that you can run in jupyterlab
-- these are far from being complete, but still demonstrate how to use the
routines:

```
contlin_demo.ipynb
finite_demo.ipynb
linear_demo.ipynb
lqr_iter_demo.ipynb
mass_demo.ipynb
nonlinear_demo.ipynb
viterbi_ge_demo.ipynb
```

Directories experiments-bellman/, experiments-viterbi/, and experiments-hjb contain scripts for running speed experiments. They would work just like:

```
% cd experiments-bellman
% mkdir -p res
% sh cpu_tracking_1.sh
```

# Related papers

The following papers are closely related to the present ones:

*[4] Sakira Hassan, Simo Särkkä, Ángel F. García-Fernández (2021). Temporal Parallelization of Inference in Hidden Markov Models. IEEE Transactions on Signal Processing, Volume 69, Pages 4875-4887.*

- Open access: https://doi.org/10.1109/TSP.2021.3103338
- arXiv: https://arxiv.org/abs/2102.05743
- Code: https://github.com/EEA-sensors/sequential-parallelization-examples/tree/main/python/temporal-parallelization-inference-in-HMMs

*[5] Simo Särkkä and Ángel F. García-Fernández (2021). Temporal Parallelization of Bayesian Smoothers. IEEE Transactions on Automatic Control, Volume 66, Issue 1, Pages 299-306.*

- arXiv: https://arxiv.org/abs/1905.13002
- Code: https://github.com/EEA-sensors/sequential-parallelization-examples/tree/main/python/temporal-parallelization-bayes-smoothers

Furthermore, a good source for all kinds of classical filters and smoothers is the following textbook:

*[6] Simo Särkkä and Lennart Svensson (2023). Bayesian Filtering and Smoothing, Second Edition. Cambridge University Press.*

- Open access: http://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf
- Code: https://github.com/EEA-sensors/Bayesian-Filtering-and-Smoothing


# Exercise

Kalman filters, RTS smoothers, and LQT regulators are closely related in two ways:

1. Kalman filter and RTS smoother are "duals" of LQT.
2. Kalman filter and RTS smoother can be derived as a special case of Viterbi algorithm for a linear Gaussian state-space model.

*Exercise:* Use the parallel LQT codes in this repository to implement parallel Kalman filters and smoothers introduced in [5] using one or both of the points of view above.
