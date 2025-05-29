# SAMPa: Sharpness-aware Minimization Parallelized

This is the official code for [SAMPa: Sharpness-aware Minimization Parallelized](https://arxiv.org/pdf/2410.10683v1) accepted at NeurIPS 2024.

SAMPa introduces a fully parallelized version of sharpness-aware minimization (SAM) by allowing the two gradient computations to occur simultaneously:

$$
\begin{aligned}
\widetilde{x}_t &= x\_t + \rho \frac{\nabla f(y\_t)}{\lVert \nabla f(y\_t) \rVert} \\
y\_{t+1} &= x\_t - \eta\_t  \nabla f(y\_t) \\
x\_{t+1} &= x\_t - \eta\_t (1-\lambda) \nabla f (\widetilde{x}\_t) - \eta\_t \lambda \nabla f(y\_{t+1})
\end{aligned}
$$


where the gradients $\nabla f(\widetilde{x}_t)$ and $\nabla f(y\_{t+1})$ are computed in parallel, significantly improving efficiency.

SAMPa serves as one of the most efficient SAM variants:

<img src="./figs/SAMPa_numGrads.png" width="300"> <img src="./figs/SAMPa_Time.png" width="300">

## Setup

```
conda create -n sampa python=3.8
conda activate sampa

# On GPU
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r requirements.txt
```


## Usage

This code is for SAMPa's implementation. It parallelizes two gradient computations on 2 GPUs. 
Specifically in `train.py`, `global_rank:0` handles $\nabla f (\widetilde{x}_t)$ and `global_rank:1` handles $\nabla f(y\_{t+1})$.

To train ResNet-56 on CIFAR-10 using SAMPa, use the following command:

```
CUDA_VISIBLE_DEVICES=0,1 python train.py --model resnet56 --dataset cifar10 --rho 0.1 --epochs 200
```


## Citation
```
@inproceedings{xie2024sampa,
  title={{SAMPa}: Sharpness-aware Minimization Parallelized},
  author={Xie, Wanyun and Pethick, Thomas and Cevher, Volkan},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```
