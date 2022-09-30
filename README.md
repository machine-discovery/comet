# Constants of motion network (COMET)

This is the repository accompanying the paper "Constants of motion network".

https://user-images.githubusercontent.com/1624640/165353470-5b43fc1a-77de-4705-98e8-2dde532119b2.mp4

## Getting started

To ensure reproducibility, please install the exact version in the requirements.
If you are using conda, you can follow:

```
conda create -n comet python=3.9
conda activate comet
```
Then follow the instruction in pytorch.org to install pytorch 1.11.0, then you can run:
```
python -m pip install -r requirements.txt
```

## Orthogonalization part in the code

If you come only to see the orthogonalization code, take a look at the `methods.py`, under
the object `CoMet` and the method `forward` and follow the branches where `ncom != 0`.

## Files guide

Files that can be executed:

* `main.py`: the training file
* `calc_mse.py`: the file to calculate MSE (mean squared error)
* `calc_com.py`: the file to plot the constants of motion values for different cases and methods
* `calc_ncom.py`: the file to plot the average loss L1 values for different number of constants of motion
* `calc_ncom2.py`: the file to plot the L1 values during the training
* `vis_cont.py`: the file to plot the end state of continuous states simulation

Those files have options that can be set.
To see the option, you can add `--help`, for example: `python main.py --help`

The helper files are:

* `methods.py`: list the deep learning methods that we run
* `sims.py`: list the simulations that we run

## How to replicate the results on the paper

#### Section 4: learning constants of motion from data

To run the experiments:

```
python main.py --method node --sim mass-spring --version 1
python main.py --method hnn --sim mass-spring --version 2
python main.py --method nsf --sim mass-spring --version 3
python main.py --method lnn --sim mass-spring --version 4
python main.py --method comet --sim mass-spring --version 5
python main.py --method comet --ncom 1 --sim mass-spring --version 6
python main.py --method comet --ncom 2 --sim mass-spring --version 7

python main.py --method node --sim 2d-pendulum --version 11
python main.py --method hnn --sim 2d-pendulum --version 12
python main.py --method nsf --sim 2d-pendulum --version 13
python main.py --method lnn --sim 2d-pendulum --version 14
python main.py --method comet --sim 2d-pendulum --version 15
python main.py --method comet --ncom 1 --sim 2d-pendulum --version 16
python main.py --method comet --ncom 2 --sim 2d-pendulum --version 17

python main.py --method node --sim damped-pendulum --version 21
python main.py --method hnn --sim damped-pendulum --version 22
python main.py --method nsf --sim damped-pendulum --version 23
python main.py --method lnn --sim damped-pendulum --version 24
python main.py --method comet --sim damped-pendulum --version 25
python main.py --method comet --ncom 1 --sim damped-pendulum --version 26
python main.py --method comet --ncom 2 --sim damped-pendulum --version 27

python main.py --method node --sim two-body --version 31
python main.py --method hnn --sim two-body --version 32
python main.py --method nsf --sim two-body --version 33
python main.py --method lnn --sim two-body --version 34
python main.py --method comet --sim two-body --version 35
python main.py --method comet --ncom 1 --sim two-body --version 36
python main.py --method comet --ncom 2 --sim two-body --version 37
python main.py --method comet --ncom 3 --sim two-body --version 38
python main.py --method comet --ncom 4 --sim two-body --version 39
python main.py --method comet --ncom 5 --sim two-body --version 40
python main.py --method comet --ncom 6 --sim two-body --version 30

python main.py --method node --sim nonlin-spring-2d --version 41
python main.py --method hnn --sim nonlin-spring-2d --version 42
python main.py --method nsf --sim nonlin-spring-2d --version 43
python main.py --method lnn --sim nonlin-spring-2d --version 44
python main.py --method comet --sim nonlin-spring-2d --version 45
python main.py --method comet --ncom 1 --sim nonlin-spring-2d --version 46
python main.py --method comet --ncom 2 --sim nonlin-spring-2d --version 47

python main.py --method node --sim lotka-volterra --version 51
python main.py --method hnn --sim lotka-volterra --version 52
python main.py --method nsf --sim lotka-volterra --version 53
python main.py --method lnn --sim lotka-volterra --version 54
python main.py --method comet --sim lotka-volterra --version 55
python main.py --method comet --ncom 1 --sim lotka-volterra --version 56
python main.py --method comet --ncom 2 --sim lotka-volterra --version 57
```

To get the RMSE numbers on Table 2:

```
python calc_mse.py --method node --sim mass-spring --version 1
python calc_mse.py --method hnn --sim mass-spring --version 2
python calc_mse.py --method nsf --sim mass-spring --version 3
python calc_mse.py --method lnn --sim mass-spring --version 4
python calc_mse.py --method comet --sim mass-spring --version 5
python calc_mse.py --method comet --ncom 1 --sim mass-spring --version 6
python calc_mse.py --method comet --ncom 2 --sim mass-spring --version 7

python calc_mse.py --method node --sim 2d-pendulum --version 11
python calc_mse.py --method hnn --sim 2d-pendulum --version 12
python calc_mse.py --method nsf --sim 2d-pendulum --version 13
python calc_mse.py --method lnn --sim 2d-pendulum --version 14
python calc_mse.py --method comet --sim 2d-pendulum --version 15
python calc_mse.py --method comet --ncom 1 --sim 2d-pendulum --version 16
python calc_mse.py --method comet --ncom 2 --sim 2d-pendulum --version 17

python calc_mse.py --method node --sim damped-pendulum --version 21
python calc_mse.py --method hnn --sim damped-pendulum --version 22
python calc_mse.py --method nsf --sim damped-pendulum --version 23
python calc_mse.py --method lnn --sim damped-pendulum --version 24
python calc_mse.py --method comet --sim damped-pendulum --version 25
python calc_mse.py --method comet --ncom 1 --sim damped-pendulum --version 26
python calc_mse.py --method comet --ncom 2 --sim damped-pendulum --version 27

python calc_mse.py --method node --sim two-body --version 31
python calc_mse.py --method hnn --sim two-body --version 32
python calc_mse.py --method nsf --sim two-body --version 33
python calc_mse.py --method lnn --sim two-body --version 34
python calc_mse.py --method comet --sim two-body --version 35
python calc_mse.py --method comet --ncom 1 --sim two-body --version 36
python calc_mse.py --method comet --ncom 2 --sim two-body --version 37
python calc_mse.py --method comet --ncom 3 --sim two-body --version 38
python calc_mse.py --method comet --ncom 4 --sim two-body --version 39
python calc_mse.py --method comet --ncom 5 --sim two-body --version 40
python calc_mse.py --method comet --ncom 6 --sim two-body --version 30

python calc_mse.py --method node --sim nonlin-spring-2d --version 41
python calc_mse.py --method hnn --sim nonlin-spring-2d --version 42
python calc_mse.py --method nsf --sim nonlin-spring-2d --version 43
python calc_mse.py --method lnn --sim nonlin-spring-2d --version 44
python calc_mse.py --method comet --sim nonlin-spring-2d --version 45
python calc_mse.py --method comet --ncom 1 --sim nonlin-spring-2d --version 46
python calc_mse.py --method comet --ncom 2 --sim nonlin-spring-2d --version 47

python calc_mse.py --method node --sim lotka-volterra --version 51
python calc_mse.py --method hnn --sim lotka-volterra --version 52
python calc_mse.py --method nsf --sim lotka-volterra --version 53
python calc_mse.py --method lnn --sim lotka-volterra --version 54
python calc_mse.py --method comet --sim lotka-volterra --version 55
python calc_mse.py --method comet --ncom 1 --sim lotka-volterra --version 56
python calc_mse.py --method comet --ncom 2 --sim lotka-volterra --version 57
```

To get the discovered constants of motion plot (Figure 1):

```
python calc_com_contour.py --sim mass-spring --version 5
python calc_com_contour.py --sim lotka-volterra --version 55
```

To get the constants of motion plot (Figure 2):

```
python calc_com.py --sim mass-spring --methods node hnn nsf lnn comet --versions 1 2 3 4 5 --figname com-mass-spring.png
python calc_com.py --sim 2d-pendulum --methods node hnn nsf lnn comet --versions 11 12 13 14 15 --figname com-2d-pendulum.png
python calc_com.py --sim two-body --methods node hnn nsf lnn comet --versions 31 32 33 34 35 --figname com-two-body.png
```

To get the trajectory of the two body in Figure 3:

```
python calc_two_body.py --methods node hnn nsf lnn comet --versions 31 32 33 34 35 --figname two-body-trajectory.png
```

#### Section 5: systems with external influence

To run the experiment on the forced pendulum:

```
python main.py --method comet --ncom 0 --sim forced-2d-pendulum --version 91
python main.py --method comet --ncom 3 --sim forced-2d-pendulum --version 92
```

Then, to get the figure 4:

```
python calc_com.py --sim forced-2d-pendulum --methods comet comet --ncoms 0 3 --versions 91 92 --figname com-forced-pendulum-2d.png
```

#### Section 6: finding the number of constants of motion

This is the longest experiment for this paper, to run them:

```
python main.py --method node --noise 0.0 --sim damped-pendulum --version 100 --seed 123
python main.py --method comet --noise 0.0 --ncom 1 --sim damped-pendulum --version 101 --seed 123
python main.py --method comet --noise 0.0 --ncom 2 --sim damped-pendulum --version 102 --seed 123
python main.py --method comet --noise 0.0 --ncom 3 --sim damped-pendulum --version 103 --seed 123

python main.py --method node --noise 0.0 --sim two-body --version 110 --seed 123
python main.py --method comet --noise 0.0 --ncom 1 --sim two-body --version 111 --seed 123
python main.py --method comet --noise 0.0 --ncom 2 --sim two-body --version 112 --seed 123
python main.py --method comet --noise 0.0 --ncom 3 --sim two-body --version 113 --seed 123
python main.py --method comet --noise 0.0 --ncom 4 --sim two-body --version 114 --seed 123
python main.py --method comet --noise 0.0 --ncom 5 --sim two-body --version 115 --seed 123
python main.py --method comet --noise 0.0 --ncom 6 --sim two-body --version 116 --seed 123
python main.py --method comet --noise 0.0 --ncom 7 --sim two-body --version 117 --seed 123

python main.py --method node --noise 0.0 --sim nonlin-spring-2d --version 120 --seed 123
python main.py --method comet --noise 0.0 --ncom 1 --sim nonlin-spring-2d --version 121 --seed 123
python main.py --method comet --noise 0.0 --ncom 2 --sim nonlin-spring-2d --version 122 --seed 123
python main.py --method comet --noise 0.0 --ncom 3 --sim nonlin-spring-2d --version 123 --seed 123
```
Then repeat with `--seed` value to be `122`, `121`, `120`, `119`, keeping the version number the same.

To get the figure 5 a-c, run:

```
python calc_ncom.py --sim damped-pendulum --versions 100 101 102 103 --figname ncom-damped-pendulum.png
python calc_ncom.py --sim two-body --versions 110 111 112 113 114 115 116 117 --figname  ncom-two-body.png
python calc_ncom.py --sim nonlin-spring-2d --versions 120 121 122 123 --figname ncom-nonlin-spring-2d.png
```

To get the figure 5 d-e, save the `val_loss` csv from the tensorboard with one of the seed values in a folder
`trainings_ncom/{sim_name}` where `{sim_name}` is the argument after `--sim`, then run

```
python calc_ncom2.py --sim damped-pendulum --figname ncom2-damped-pendulum.png
python calc_ncom2.py --sim two-body --figname ncom2-two-body.png
python calc_ncom2.py --sim nonlin-spring-2d --figname ncom2-nonlin-spring-2d.png
```

To run the failure mode, run:

```
python main.py --method comet-50 --ncom 0 --noise 0.0 --sim nonlin-spring-2d --version 150
python main.py --method comet-50 --ncom 1 --noise 0.0 --sim nonlin-spring-2d --version 151
python main.py --method comet-50 --ncom 2 --noise 0.0 --sim nonlin-spring-2d --version 152
python main.py --method comet-50 --ncom 3 --noise 0.0 --sim nonlin-spring-2d --version 153
```

#### Section 7: simulation with infinite number of states

To run the experiment:

```
python main.py --method cometcont --sim kdvcont --noise 0.0 --ncom 0 --version 200
python main.py --method cometcont --sim kdvcont --noise 0.0 --ncom 1 --version 201
python main.py --method cometcont --sim kdvcont --noise 0.0 --ncom 2 --version 202
```

To get the figure 7, run:

```
python vis_cont.py --sim kdvcont --ncoms 0 1 2 --versions 200 201 202 --figname cont-res-kdv.png
```

## Appendix: ablation study on the noise level

For noise 0.1:

```
python main.py --method node --sim mass-spring --version 301 --noise 0.1
python main.py --method hnn --sim mass-spring --version 302 --noise 0.1
python main.py --method nsf --sim mass-spring --version 303 --noise 0.1
python main.py --method lnn --sim mass-spring --version 304 --noise 0.1
python main.py --method comet --sim mass-spring --version 305 --noise 0.1

python main.py --method node --sim 2d-pendulum --version 311 --noise 0.1
python main.py --method hnn --sim 2d-pendulum --version 312 --noise 0.1
python main.py --method nsf --sim 2d-pendulum --version 313 --noise 0.1
python main.py --method lnn --sim 2d-pendulum --version 314 --noise 0.1
python main.py --method comet --sim 2d-pendulum --version 315 --noise 0.1

python main.py --method node --sim damped-pendulum --version 321 --noise 0.1
python main.py --method hnn --sim damped-pendulum --version 322 --noise 0.1
python main.py --method nsf --sim damped-pendulum --version 323 --noise 0.1
python main.py --method lnn --sim damped-pendulum --version 324 --noise 0.1
python main.py --method comet --sim damped-pendulum --version 325 --noise 0.1

python main.py --method node --sim two-body --version 331 --noise 0.1
python main.py --method hnn --sim two-body --version 332 --noise 0.1
python main.py --method nsf --sim two-body --version 333 --noise 0.1
python main.py --method lnn --sim two-body --version 334 --noise 0.1
python main.py --method comet --sim two-body --version 335 --noise 0.1

python main.py --method node --sim nonlin-spring-2d --version 341 --noise 0.1
python main.py --method hnn --sim nonlin-spring-2d --version 342 --noise 0.1
python main.py --method nsf --sim nonlin-spring-2d --version 343 --noise 0.1
python main.py --method lnn --sim nonlin-spring-2d --version 344 --noise 0.1
python main.py --method comet --sim nonlin-spring-2d --version 345 --noise 0.1

python main.py --method node --sim lotka-volterra --version 351 --noise 0.1
python main.py --method hnn --sim lotka-volterra --version 352 --noise 0.1
python main.py --method nsf --sim lotka-volterra --version 353 --noise 0.1
python main.py --method lnn --sim lotka-volterra --version 354 --noise 0.1
python main.py --method comet --sim lotka-volterra --version 355 --noise 0.1
```

For noise 0.2:

```
python main.py --method node --sim mass-spring --version 401 --noise 0.2
python main.py --method hnn --sim mass-spring --version 402 --noise 0.2
python main.py --method nsf --sim mass-spring --version 403 --noise 0.2
python main.py --method lnn --sim mass-spring --version 404 --noise 0.2
python main.py --method comet --sim mass-spring --version 405 --noise 0.2

python main.py --method node --sim 2d-pendulum --version 411 --noise 0.2
python main.py --method hnn --sim 2d-pendulum --version 412 --noise 0.2
python main.py --method nsf --sim 2d-pendulum --version 413 --noise 0.2
python main.py --method lnn --sim 2d-pendulum --version 414 --noise 0.2
python main.py --method comet --sim 2d-pendulum --version 415 --noise 0.2

python main.py --method node --sim damped-pendulum --version 421 --noise 0.2
python main.py --method hnn --sim damped-pendulum --version 422 --noise 0.2
python main.py --method nsf --sim damped-pendulum --version 423 --noise 0.2
python main.py --method lnn --sim damped-pendulum --version 424 --noise 0.2
python main.py --method comet --sim damped-pendulum --version 425 --noise 0.2

python main.py --method node --sim two-body --version 431 --noise 0.2
python main.py --method hnn --sim two-body --version 432 --noise 0.2
python main.py --method nsf --sim two-body --version 433 --noise 0.2
python main.py --method lnn --sim two-body --version 434 --noise 0.2
python main.py --method comet --sim two-body --version 435 --noise 0.2

python main.py --method node --sim nonlin-spring-2d --version 441 --noise 0.2
python main.py --method hnn --sim nonlin-spring-2d --version 442 --noise 0.2
python main.py --method nsf --sim nonlin-spring-2d --version 443 --noise 0.2
python main.py --method lnn --sim nonlin-spring-2d --version 444 --noise 0.2
python main.py --method comet --sim nonlin-spring-2d --version 445 --noise 0.2

python main.py --method node --sim lotka-volterra --version 451 --noise 0.2
python main.py --method hnn --sim lotka-volterra --version 452 --noise 0.2
python main.py --method nsf --sim lotka-volterra --version 453 --noise 0.2
python main.py --method lnn --sim lotka-volterra --version 454 --noise 0.2
python main.py --method comet --sim lotka-volterra --version 455 --noise 0.2
```
