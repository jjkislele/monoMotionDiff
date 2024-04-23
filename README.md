# Exploring Latent Cross-Channel Embedding for Accurate 3D Human Pose Reconstruction in a Diffusion Framework

[Junkun Jiang](https://scholar.google.com/citations?user=Oce7VfUAAAAJ&hl=en), and [Jie Chen](https://scholar.google.com/citations?user=qrWi1RYAAAAJ&hl=en)\*, Hong Kong Baptist University

\* *Corresponding author*

---

[Paper](https://arxiv.org/abs/2211.16940) | [Project Page](https://jjkislele.github.io/pages/projects/monoMotionDiff/) | [BU-MCV lab](https://bumcv.github.io/) | [HKBU-VSComputing](https://github.com/HKBU-VSComputing)

---

<p align="center"> <img src="https://github.com/jjkislele/jjkislele.github.io/blob/master/pages/projects/monoMotionDiff/assets/framework.jpg?raw=true" width="100%"> </p>

## How to deploy

### Dependencies

The code is tested on Windows with

```
pytorch                   1.10.2
torchvision               0.11.3
CUDA                      11.3.1
```

We suggest using the virtual environment and an easy-to-use package/environment manager such as conda to maintain the project.

```python
conda create -n icassp python=3.6
conda activate icassp
# install pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# install the rest of the dependencies
pip install -r requirements.txt
```

### Dataset

Following [DiffPose](https://github.com/GONGJIA0208/Diffpose), we utilize the GMM-fitted pose data as input during training and testing. Please use this [link](https://www.dropbox.com/sh/54lwxf9zq4lfzss/AABmpOzg31PrhxzcxmFQt3cYa?dl=0) provided by [DiffPose](https://github.com/GONGJIA0208/Diffpose) to download the data. Please put those npz files into the `./data` directory.

Here are explanations of the input data:

```python
./data/data_2d_h36m_cpn_ft_h36m_dbb_gmm.npz  # 2D estimated poses sampled from a GMM
./data/data_2d_h36m_gt_gmm.npz               # 2D ground-truth poses sampled from a GMM
./data/data_3d_h36m.npz                      # 3D ground-truth poses
```

#### Prepare 2D-to-3D lifter

The pretrained 2D-to-3D lifting model can be downloaded from the following table. All weights come from [DiffPose](https://github.com/GONGJIA0208/Diffpose).

| Name         | Description        | URL |
|--------------|--------------------|-----|
| gcn_xyz_cpn.pth | Trained on 2D estimated input | [link](https://www.dropbox.com/sh/jhwz3ypyxtyrlzv/AABivC5oiiMdgPePxekzu6vga?e=1&preview=gcn_xyz_cpn.pth&st=kn7xvh6k&dl=0) |
| gcn_xyz_gt.pth  | Trained on 2D gt input        | [link](https://www.dropbox.com/sh/jhwz3ypyxtyrlzv/AABivC5oiiMdgPePxekzu6vga?e=1&preview=gcn_xyz_gt.pth&st=uxt1uv0j&dl=0) |

Please put them in the folder ``ckpts``.

#### Prepare 2D normalized poses

To speed up the 2D sampling process, we prepare a simple script to normalize the sampled 2D poses to the UV space in advance. Please run the following command.

```shell
python prepare_2d_poses.py
```

### Training

To train a diffusion model from scratch, simply paste the following command to your console, after the `icassp` environment has been activated.

```shell
python train.py \
--config cfgs/cfg_cpn.yml \  # config for 2D estimated pose input
--exp exp \                  # experiment root path
--doc human36m_cpn           # the name of the folder for storing weights, config.yml, log, etc.
```

```shell
python train.py \
--config cfgs/cfg_cpn.yml \  # config for 2D ground-truth pose input
--exp exp \                  # experiment root path
--doc human36m_gt            # the name of the folder for storing weights, config.yml, log, etc.
```

### Evaluation

The pretrained diffusion model can be downloaded from the following table.

| Name         | Description        | URL |
|--------------|--------------------|-----|
| ckpt_cpn.pth | Trained on 2D estimated input | [link](https://www.dropbox.com/scl/fi/o8cp8wcliwnqfbd159u27/ckpt_cpn.pth?rlkey=cn0ze4m9dbtfwmcotmxhvhiym&st=q0zh2rqn&dl=0) |
| ckpt_gt.pth  | Trained on 2D gt input        | [link](https://www.dropbox.com/scl/fi/bndqupb1afs0iigeknex2/ckpt_gt.pth?rlkey=gagi4i4x3km5c5y76tcbdefbo&st=ckmg2enn&dl=0) |

Similarly, please put them in the folder ``ckpts`` and run the following command.

```shell
python eval.py \
--config cfgs/cfg_cpn.yml \  # config for 2D estimated pose input
--exp exp \                  # experiment root path
--doc human36m_cpn           # the name of the folder for storing weights, config.yml, log, etc.
```

```shell
python eval.py \
--config cfgs/cfg_gt.yml \   # config for 2D ground-truth pose input
--exp exp \                  # experiment root path
--doc human36m_gt            # the name of the folder for storing weights, config.yml, log, etc.
```

The results will be displayed in the console like:

```shell
===Action=== ==p#1 mm== =p#2 mm=
Directions    43.33      34.59
...
Average       49.40      39.05
```

## Bibtex

If you use our code/models in your research, please cite our paper :raised_hands: :

```
@inproceedings{jiang2024diff,
  title={Exploring Latent Cross-Channel Embedding for Accurate 3d Human Pose Reconstruction in a Diffusion Framework},
  author={Jiang, Junkun and Chen, Jie},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7870-7874},
  doi={10.1109/ICASSP48485.2024.10448487},
  year={2024}
}
```

## Acknowledgement

Many thanks to the following open-source repositories for their help in developing our project.

- The diffusion learning-based monocular 3D pose estimation [DiffPose](https://github.com/GONGJIA0208/Diffpose). We thank them for their great work :heart:. The main structure is built on it.
- The GCN backbone [Graformer](https://github.com/zhaoweixi/GraFormer).
- The evaluation code from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D).
- The diffusion pipeline from [DDIM](https://github.com/ermongroup/ddim).
