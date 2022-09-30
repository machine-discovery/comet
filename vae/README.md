```
python main.py --method comet --sim two-body --nstates 10 --ncom 9 --version 0
python main.py --method comet --sim two-body --nstates 10 --ncom 0 --version 1
python main.py --method hnn --sim two-body --nstates 10 --ncom 9 --version 2
python main.py --method nsf --sim two-body --nstates 10 --ncom 9 --version 3
```

```
python show_trajectory.py --sim two-body --methods comet comet hnn nsf --nstates 10 10 10 10 --ncoms 9 0 0 0 --versions 0 1 2 3 --names COMET NODE HNN NSF
```
