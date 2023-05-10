# Server initial model trained only with partial1

## Installation of dependencies

To keep the git repository with a minimal size, you'll have to download the following two
parts manually:

- [anomalib](https://github.com/openvinotoolkit/anomalib/archive/refs/tags/v0.4.0.tar.gz) - 
to be installed in ./anomalib
- [static](https://www.mvtec.com/company/research/datasets/mvtec-ad) - 
sign up and download the dataset - or use the links below

For the `static` directory, the structure must be:
- `static`
  - `full-mvtec`
    - [carpet](https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951672/carpet.tar.xz) 
    - [grid](https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937487-1629951814/grid.tar.xz)
    - [leather](https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937607-1629951964/leather.tar.xz)
  - `all-partials-mvtec`
    - `partial1` - copy of `carpet`, `grid`, `leather`, but for the `*/train/good` directory only the files 
    `000.png` to `100.png` 
    - `partial2` - copy of `carpet`, `grid`, `leather`, but for the `*/train/good` directory only the files 
    `101.png` to `200.png` 
    - `partial3` - copy of `carpet`, `grid`, `leather`, but for the `*/train/good` directory only the files 
    `201.png` to the last file 

## Patchcore 
Tested with python 3.10.7

Setup env:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Train patchcore based on config:

```
python anomalib/tools/train.py --config config-generic.yaml
```

Here are the relevant parameters you can modify in `config-generic.yaml`:

* `dataset.path`: where your input dataset is
  * In our case, I've already separated the dataset in multiple parts:
    * `static/full-mvtec`: full dataset
    * `static/all-partials-mvtec/partial1`
    * `static/all-partials-mvtec/partial2`
    * `static/all-partials-mvtec/partial3`
    * `static/all-partials-mvtec/small-partial`
* `dataset.category`: a folder inside the `path`
  * `carpet`
  * `grid`
  * `leather`
* `model.coreset_sampling_ratio`: percentage of the extracted feature to reduce to
  * TODO: experiment to see if this affect the results noticeably
* `project.path`: where to save the trained model and test results
* `trainer.accelerator`: if you are able to run on a GPU
* You can tweak other parameter to affect the anomalib process if needed too.

Once you have trained a model, you can check the size of the memory bank with this script:

```
python test.py --config config-generic.yaml --weights static/results/patchcore/mvtec/grid/run/weights/model.ckpt
```

You can test your model to get the metrics (need the correct dataset format with the `ground_truth` `test` and `train` folders):

```
python anomalib/tools/test.py --model patchcore --config config-generic.yaml --weight_file static/results/patchcore/mvtec/grid/run/weights/model.ckpt
```

You can also merge memory banks with this script:

```
python manual-merge.py --config config-partial2.yaml --weights <path to ckpt1> <path to ckpt2>
```

If you check the memory bank size of the resulting `merged.ckpt` you will see that it has increased by concatenating the memory banks.

If you need to understand more about the underlying implementation of Patchcore by Anomalib, take a look at these 2 files configuring the training:

* `anomalib/src/anomalib/models/patchcore/torch_model.py`
* `anomalib/src/anomalib/models/patchcore/lightning_model.py`

Anomalib leverage [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) for easier training configuration.

## Aggregation server POC

I've started working on a small poc to run a server that could aggregate memory bank from different client, but it's not correctly working yet.

```
flask --app server run --port=8080
```

```
flask --app client run
```
