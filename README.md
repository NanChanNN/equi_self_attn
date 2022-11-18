# Equivariant Self-attention via Invariant Features

**Introduction**

This is a repository of the official implementation of paper *Equivariant self-attention via invariant features*. Module implementation is placed in `modules/attn_modules.py`.


**Prerequisites**

- [Pytorch](https://pytorch.org/) 1.12.0
- [DGL](https://www.dgl.ai/) 0.9.0
- `requirements.txt` includes all the dependencies required.



**Steps to run the NBody experiment**:

1)  Create a dataset using scripts in the folder `NBody/data_generation` with the following commands: 

   ```shell
   python3 generate_dataset.py --num-train 5000 --num-test 1000 --n-balls 5
   ```

2) Run the experiment using the following commands: 
    For OD model:
   ```shell
   python nbody_run.py --batch_size 128 --num_channels 5 --num_layers 4 --data_str 5_new --model MyModel_OD
   ```
    For SOD model:
   ```shell
   python nbody_run.py --batch_size 128 --num_channels 4 --num_layers 4 --data_str 5_new --model MyModel_SOD
   ```


**Steps to run the QM9 experiment**:

1) Download a preprocessed QM9 dataset [here](https://drive.google.com/file/d/1EpJG0Bo2RPK30bMKK6IUdsR5r0pTBEP0/view?usp=sharing) and put it into `qm9`

2) Run the experiment using the following commands (use task *homo* as an example):
   For OD model: 
   ```shell
   python qm9_run.py --task homo --batch_size 128 --num_epochs 200 --num_layers 7 --div 2 --head 8 --pooling sum --model MyModel_OD
   ```
   For SOD model:
   ```shell
   python qm9_run.py --task homo --batch_size 128 --num_epochs 200 --num_layers 7 --div 2 --head 8 --pooling sum --model MyModel_SOD
   ```
    
3) To evaluate the model on other tasks, set `--task` option to one of `['homo, 'mu', 'alpha', 'lumo', 'gap', 'cv']`.

   

**Credit to 'SE3-Transformer'**

The scripts of our NBody and QM9 experiments (in subfolder `NBody/` and `QM9/`) are strongly based on https://github.com/FabianFuchsML/se3-transformer-public, which is under the MIT license. It consists of the official implementation of the following paper:

```
@inproceedings{fuchs2020se3transformers,
    title={SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks},
    author={Fabian B. Fuchs and Daniel E. Worrall and Volker Fischer and Max Welling},
    year={2020},
    booktitle = {Advances in Neural Information Processing Systems 34 (NeurIPS)},
}
```

