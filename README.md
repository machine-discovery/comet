# Constants of motion network (COMET)

This is the repository accompanying the paper "Constants of motion network".
All of the codes and notes are in the `contents` folder.

![two-body-animation-smaller](https://user-images.githubusercontent.com/1624640/193280954-6e275aea-b0a3-4091-ba1f-942a69c77d38.gif)

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
Or you can also follow the simplistic implementation below (only 30 lines of code).

```python
import torch
import functorch

class COMET(torch.nn.Module):
    def __init__(self, nstates: int, ncom: int):
        super().__init__()
        assert ncom < nstates
        self._nstates = nstates
        self._nn = torch.nn.Sequential(
            torch.nn.Linear(nstates, 250), torch.nn.LogSigmoid(),
            torch.nn.Linear(250, 250), torch.nn.LogSigmoid(),
            torch.nn.Linear(250, 250), torch.nn.LogSigmoid(),
            torch.nn.Linear(250, nstates + ncom))
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # states: (batch_size, nstates)
        # returns dstates/dt prediction with shape: (batch_size, nstates)
        states = states.requires_grad_()
        def _get_com_dstates(states):
            # states: (nstates,)
            nnout = self._nn(states)  # (nstates + ncom,)
            dstates, com = torch.split(nnout, split_size_or_sections=self._nstates, dim=-1)
            return com, (dstates,)
        jac_fcn = functorch.vmap(functorch.jacrev(_get_com_dstates, 0, has_aux=True))
        dcom_jac, (dstates,) = jac_fcn(states)
        dcom_jac = dcom_jac.transpose(-2, -1).contiguous()
        dcom_jac_dstates = torch.cat((dcom_jac, dstates[..., None]), dim=-1)
        q, r = torch.linalg.qr(dcom_jac_dstates)
        dstates_ortho = q[..., -1] * r[..., -1, -1:]
        return dstates_ortho
```

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

The files for the learning from pixel experiment are contained in the `vae` folder.

## How to replicate the results on the paper

If you want to run every single experiment in the paper, we have enlisted the commands we used in the file `commands-for-replication.md`.
