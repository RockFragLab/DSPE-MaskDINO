# DSPE-MaskDINO for Rock-Fragment Instance Segmentation

Research code for instance segmentation of rock fragments in blast-pile images. The model extends [MaskDINO](https://github.com/IDEA-Research/MaskDINO) with a ConvNeXtV2-Tiny backbone, improved positional encoding, and a Directional-SCSA feature enhancement module. 
## Environment

The experiments were run with Python 3.9.25, PyTorch 2.7.0 + CUDA 12.8, torchvision 0.22.0 + CUDA 12.8, and a locally built Detectron2 0.6. Other Python packages are listed in `requirements.txt`. CUDA extension compilation also requires a compatible CUDA toolkit and C++ compiler.

Install PyTorch and torchvision using the matching CUDA build from the [official PyTorch installation guide](https://pytorch.org/get-started/previous-versions/). Then install a compatible [Detectron2 source build](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) and the remaining packages:

```bash
python -m pip install -r requirements.txt
```

Build the multi-scale deformable attention extension from the repository root after confirming that `CUDA_HOME` points to the installed CUDA toolkit:

```bash
cd maskdino/modeling/pixel_decoder/ops
python setup.py build install
cd ../../../..
```

The extension build is platform-dependent. See the [upstream MaskDINO installation guide](https://github.com/IDEA-Research/MaskDINO/blob/main/INSTALL.md) and verify that the installed Detectron2, PyTorch, CUDA toolkit, and compiler are compatible before training.

## Data

The dataset registration code is `maskdino/data/datasets/register_date_coco.py`. Set `DSPE_DATE1_ROOT` and `DSPE_DATE2_ROOT` to the respective dataset directories. If unset, the code looks under `datasets/date1` and `datasets/date2` relative to the repository root. Dataset files are not included in this repository.

For example, in PowerShell:

```powershell
$env:DSPE_DATE1_ROOT = "D:\path\to\date1"
$env:DSPE_DATE2_ROOT = "D:\path\to\date2"
```

Each dataset is expected to have the following COCO-format layout:

```text
<dataset-root>/
  train/
    _annotations.coco.json
    <images>
  valid/
    _annotations.coco.json
    <images>
  test/
    _annotations.coco.json
    <images>
```

The code registers `date1_train`, `date1_val`, `date1_test`, and the corresponding `date2_*` splits. Source category `stone` is remapped to the internal name `shone` (the spelling used by the existing training/evaluation code); `basketball` is the second class. Keep this mapping consistent when reproducing per-class metrics. Do not publish images, annotations, or third-party data without checking their redistribution terms.

The original research datasets are not publicly released with this repository. The mixed-source dataset also contains third-party imagery, which is not republished here. The code can be run with separately obtained, appropriately licensed COCO-format datasets using the layout above, but the paper's exact numerical results cannot be independently reproduced without access to the original data and splits. Dataset access should not be assumed or described as available on request unless permission has been confirmed.

## Training

Run from the repository root. The configurations below use 80 epochs and select the best checkpoint by validation-set `segm/AP-shone`:

```bash
python train_net.py --num-gpus 1 --config-file configs/coco/instance-segmentation/yaml/name.yaml
```

These configurations reference `weights/convnextv2_tiny.pth` for backbone initialization. Obtain the corresponding ConvNeXtV2-Tiny pretrained checkpoint separately and place it at that path, or update `MODEL.CONVNEXT.WEIGHTS` to its location. Do not confuse this initialization checkpoint with a trained DSPE-MaskDINO model checkpoint.

## Evaluation

Set `MODEL.WEIGHTS` to the checkpoint to evaluate. With the supplied configuration, `DATASETS.TEST` points to the **validation** split:

```bash
python train_net.py --num-gpus 1 --eval_only --config-file configs/coco/instance-segmentation/yaml/maskdino_ConvNeXtV2T_date1_dspe_80ep_best_stoneap.yaml MODEL.WEIGHTS path/to/date1_checkpoint.pth
```

To evaluate the held-out test split, use a copy of the relevant configuration with `DATASETS.TEST` changed to `("date1_test",)` or `("date2_test",)`. Keep model selection on the validation split; do not select checkpoints using test results.

The best-checkpoint configurations use the standard COCO evaluator because `TEST.BEST_CHECKPOINT_METRIC` is set. For the repository's additional instance metrics, use a separate evaluation configuration with `TEST.BEST_CHECKPOINT_METRIC: ""` and the intended `DATASETS.TEST` split. Record the checkpoint filename, dataset split, and metric configuration with every reported result.

## Apparent fragmentation analysis

`tools/compute_fragmentation.py` runs sliding-window instance segmentation and estimates apparent rock-fragment sizes from the predicted stone masks and a basketball marker of known diameter. Use your measured marker diameter in the selected unit; `246` below is an example, not a universal setting. Run from the repository root:

```bash
python tools/compute_fragmentation.py --image path/to/image.jpg --config configs/coco/instance-segmentation/yaml/maskdino_ConvNeXtV2T_date1_dspe_80ep_best_stoneap.yaml --weights path/to/date1_checkpoint.pth --dataset date1_val --marker-real-size 246 --unit mm --tile-size 750 --overlap 148 --gray-input --size-bin-edges "4.75,10,20,40,100,200,300,400,500,600,800" --output-dir outputs/fragmentation_example
```

The script writes per-instance sizes, marker measurements, a cumulative passing curve, size-bin percentages, percentile values, an optional expanded Swebrec fit, and a JSON summary. It can also reuse previously saved masks via `--instances-npz`; load only NPZ files you trust. For an overlay labeled with the computed sizes (with `--unit mm`), run:

```bash
python tools/draw_size_overlay.py --result-dir outputs/fragmentation_example --marker-real-size 246
```

The default `--weight-mode area` weights fragments by projected mask area. These are **2D apparent-size estimates**, dependent on segmentation quality, marker detection, and the assumed image-plane scale; they are not direct 3D sieve or mass measurements. The marker must be visible and detected in the analyzed image. Generated masks and measurement outputs are excluded from Git by `.gitignore`.

## Acknowledgments and license

This work builds on [MaskDINO](https://github.com/IDEA-Research/MaskDINO), [Detectron2](https://github.com/facebookresearch/detectron2), [ConvNeXt V2](https://github.com/facebookresearch/ConvNeXt-V2), and [SCSA](https://github.com/HZAI-ZJNU/SCSA). Retain upstream copyright notices and check the license of every external component and pretrained weight before redistribution. The MaskDINO-derived code in this repository carries the license terms in `LICENSE`.
