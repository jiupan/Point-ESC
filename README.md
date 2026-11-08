# Point-ESC

Implementation of **Point-ESC: Encoder-Driven Soft Clustering for 3D Point Cloud Pretraining**.

The pretraining model and downstream classifier are in `models/Point_ESC.py`.

~~~bash
git clone https://github.com/jiupan/Point-ESC.git
cd Point-ESC
~~~

## Requirements

Use a CUDA-enabled PyTorch environment with a matching CUDA toolkit and a C++ compiler.
The original pipeline uses PyTorch >= 1.7 and `timm==0.4.5`; the custom CUDA operators
must be built for the installed PyTorch/CUDA versions.

```bash
pip install -r requirements.txt
pip install torchvision
# Chamfer distance
cd extensions/chamfer_dist
python setup.py install --user
cd ../..
# PointNet++ FPS and gathering operators
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU KNN: install the wheel matching your PyTorch/CUDA environment,
# or build from https://github.com/unlimblue/KNN_CUDA
```

## Datasets

Set the local data paths in `cfgs/dataset_configs/` before running. The supported
training entry points below use ShapeNet-55, ModelNet40, ScanObjectNN and ShapeNetPart.
ShapeNetPart's path is passed through `--root`.

## Pretraining

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/pretrain.yaml --exp_name pretrain --seed 0
```

`cfgs/pretrain.yaml` uses 1,024 FPS-sampled points, 64 patches of 32 points, a 12-block
384-dimensional encoder with 6 heads, a 4-block decoder, and a masking ratio of 0.6.
The ShapeNet loader returns the full point cloud for GPU FPS in the runner.

The original FPS/KNN indices and shared patch mask are fixed **before** generating
the two augmented views. Both views reuse those grouping indices. Only one view is
reconstructed.

Encoder features **H generate Q** from `H_weak @ H_strong.transpose(-2, -1)`.
The implementation passes the weak view first and the strong view second.
Fig. 2's labels (strong view 1, weak view 2), this is `H2 @ H1.transpose(-2, -1)`.
Sinkhorn evaluates `exp(S / epsilon)` in log space, with three row-then-column
normalization iterations. Projected, L2-normalized features **Z predict Q** using
a row softmax with temperature 0.1, with weak-view patches as rows and strong-view
patches as columns. Only the target path stops gradients; the prediction path
updates the projector and encoder. KL is averaged over all `B * M * M` entries.

| Setting | Value |
| --- | --- |
| Sinkhorn iterations | 3 |
| Sinkhorn epsilon | 0.05 |
| Prediction temperature | 0.1 |
| Nominal ESC weight | 0.1 |
| ESC start epoch | 60 |
| ESC warm-up | 120 epochs |
| Student feature detach | False |
| Pretraining duration | 300 epochs |

Epoch indices are zero-based. ESC has zero weight before epoch 60. Its weight rises
linearly from 0 at epoch 60 to 0.1 at epoch 180, and then stays at 0.1. AdamW uses
learning rate 0.001, weight decay 0.05, batch size 128, and cosine learning-rate decay
with a separate 10-epoch learning-rate warm-up.

## Classification fine-tuning

ModelNet40:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_modelnet.yaml --finetune_model --exp_name modelnet --ckpts PATH_TO_PRETRAINED_CHECKPOINT --seed 0
```

ScanObjectNN (use `finetune_scan_objbg.yaml` or `finetune_scan_objonly.yaml` for the other splits):

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_scan_hardest.yaml --finetune_model --exp_name scan --ckpts PATH_TO_PRETRAINED_CHECKPOINT --seed 0
```

Fine-tuning loads the pretrained encoder into `PointTransformer`. The ESC projector
and reconstruction decoder are not used downstream. Classification uses the CLS token
concatenated with max-pooled patch tokens and trains the backbone and classification head.
ModelNet40 uses 1,024 points/64 patches; ScanObjectNN uses 2,048 xyz points/128 patches.
The three ScanObjectNN fine-tuning configs enable random rotation about the y-axis for Point-ESC. The local Point-MAE baseline uses random scaling and translation.

## Classification evaluation

ModelNet40 main evaluation uses one **10-view voting** evaluation per fine-tuned checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --test --vote --config cfgs/finetune_modelnet.yaml --exp_name modelnet_test --ckpts PATH_TO_FINETUNED_CHECKPOINT --seed 0
```

The ten logit vectors are averaged once. There is no repeated voting-and-maximum loop.
Without `--vote`, evaluation uses a single FPS-sampled input, as in the ablations:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --test --config cfgs/finetune_scan_hardest.yaml --exp_name scan_test --ckpts PATH_TO_FINETUNED_CHECKPOINT --seed 0
```

A test command evaluates one checkpoint. Three-seed summaries require three independently
fine-tuned checkpoints; repeated voting on one checkpoint is not a three-seed summary.

## Few-shot classification

Choose `--way` from 5/10, `--shot` from 10/20, and `--fold` from 0 through 9:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/fewshot.yaml --finetune_model --ckpts PATH_TO_PRETRAINED_CHECKPOINT --exp_name fewshot --way 5 --shot 10 --fold 0 --seed 0
```

## ShapeNetPart fine-tuning

```bash
cd segmentation
python main.py --gpu 0 --ckpts PATH_TO_PRETRAINED_CHECKPOINT --root PATH_TO_SHAPENETPART --learning_rate 0.0002 --epoch 300 --npoint 2048
```

The segmentation backbone uses 128 patches of 32 points. Its checkpoint is loaded
through the CPU so that it does not depend on the GPU numbering of the training machine.

## Implementation checks

```bash
python -m unittest discover -s tests -v
```

These CPU checks cover the ESC equations, gradient paths, shared group indices,
checkpoint mapping and evaluation dispatch. FPS/KNN/Chamfer CUDA operators are replaced
by small CPU reference operators in the tests; testing those native extensions requires
a configured CUDA environment.

## Acknowledgements

Built on [Point-MAE](https://github.com/Pang-Yatian/Point-MAE),
[Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch) and
[Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch).
