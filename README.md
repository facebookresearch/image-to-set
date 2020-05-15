## Image-to-Set Prediction
Companion code for
[L. Pineda, A. Salvador, et al.: Elucidating image-to-set prediction: An analysis of models, losses and datasets](https://arxiv.org/abs/1904.05709).

This repository contains a unified code-base to train and test strong image-to-set prediction (multi-label classification) baselines. The code comes with pre-defined train/valid/test splits for 5 datasets of increasing complexity ([Pascal VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/), [MS COCO 2014](http://cocodataset.org/#home), [ADE20k](http://groups.csail.mit.edu/vision/datasets/ADE20K/), [NUS-WIDE](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html) and [Recipe1M](http://im2recipe.csail.mit.edu/dataset)) as well as a common evaluation protocol to compare all models. The top ranked baselines across datasets are released together with the code.

If you find this code useful in your research, please consider citing with the following BibTeX entry:

```
@article{PinedaSalvador2019im2set,
  author    = {Pineda, Luis and Salvador, Amaia and Drozdzal, Michal and Romero, Adriana},
  title     = {Elucidating image-to-set prediction: An analysis of models, losses and datasets},
  journal   = {CoRR},
  volume    = {abs/1904.05709},
  year      = {2019},
  url       = {https://arxiv.org/abs/1904.05709},
  archivePrefix = {arXiv},
  eprint    = {1904.05709},
}
```

### Installation

This code uses Python 3.7.3 (Anaconda), PyTorch 1.1.0. and cuda version 10.0.130.

- Installing pytorch:
```bash
$ conda install pytorch torchvision cuda90 -c pytorch
```

- Install dependencies
```bash
$ pip install -r requirements.txt
```

### Datasets

#### Pascal VOC 2007

- Download [VOC 2007 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) and extract under ```/path/to/voc/```.
- Remember to also download the test set for evaluation.
- Fill in ```configs/datapaths.json``` with the path to voc dataset: ````"voc": "/path/to/voc/"````

#### MSCOCO 2014

- Download [MS COCO 2014](http://cocodataset.org/) and extract under path ```/path/to/coco/```.
- Fill in ```configs/datapaths.json``` with the path to coco dataset: ````"coco": "/path/to/coco/"````

#### NUSWIDE

- Download [NUSWIDE](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html) dataset and extract under ```/path/to/nuswide/```.
- Fill in ```configs/datapaths.json``` with the path to nuswide dataset: ````"nuswide": "/path/to/nuswide/"````

#### ADE20K

- Download [ADE20K Challenge data](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) and place under ```/path/to/ade20k/```.
- Fill in ```configs/datapaths.json``` with the path to ade20k dataset: ````"ade20k": "/path/to/ade20k/"````

#### Recipe1M

- Download [Recipe1M](http://im2recipe.csail.mit.edu/dataset/download) (registration required) and extract under ```/path/to/recipe1m/```.
- The contents of ```/path/to/recipe1m/``` should be the following:
```
det_ingrs.json
layer1.json
layer2.json
images/
images/train
images/val
images/test
```
- Pre-process dataset and build vocabularies with:

```bash
$ python src/utils/recipe1m_utils.py --recipe1m_path path_to_recipe1m
```
Resulting files will be stored under ```/path/to/recipe1m/preprocessed```.
- Fill in ```configs/datapaths.json``` with the path to recipe1m dataset: ````"recipe1m": "/path/to/recipe1m/"````

### Training

*Note: all python calls below must be run from* `./src`.

Checkpoints will be saved under a directory ```"<save_dir>/<dataset>/<model_name>/<image_model>/<experiment_name>/"```,  specified by ```--save_dir```, ```--dataset```, ```--model_name```, ```--image_model``` and ```--experiment_name```.

The recommended way to train the models reported in the paper is to use the JSON configuration files provided in
```configs``` folder. We have provided one configuration file for each combination of dataset, set predictor (model_name) and image backbone (image_model). The naming convention is ```configs/dataset/image_model_model_name.json```.

The following ```model_name``` are available:

- ```ff_bce```: Feed-forward model trained with binary cross-entropy loss.
- ```ff_iou```: Feed-forward model trained  with soft intersection-over-union loss.
- ```ff_td```: Feed-forward model trained with target distribution loss.
- ```ff_bce_cat```: Feed-forward model trained with binary cross-entropy loss and categorical distribution loss for cardinality prediction.
- ```ff_iou_cat```: Feed-forward model trained with soft intersection-over-union loss and categorical distribution loss for cardinality prediction.
- ```ff_td_cat```: Feed-forward model trained with target distribution loss and categorical distribution loss for cardinality prediction.
- ```ff_bce_dc```: Feed-forward model trained with binary cross-entropy loss and Dirichlet-categorical loss for cardinality prediction.
- ```lstm```: LSTM model trained with ```eos``` token for cardinality prediction.
- ```lstm_shuffle```: Same as ```lstm``` but labels are shuffled every time an image is loaded.
- ```lstmset```: LSTM model trained with ```eos``` token for cardinality prediction and pooled across time steps.
- ```tf```: Transformer model trained with ```eos``` token for cardinality prediction.
- ```tf_shuffle```: Same as ```tf``` but labels are shuffled every time an image is loaded.
- ```tf_set```: Transformer model trained with ```eos``` token for cardinality prediction and pooled across time steps.

The following ```image_model``` are available:
- ```resnet50```: Use resnet50 as image feature extractor.
- ```resnet101```: Use resnet101 as image feature extractor.
- ```resnext101_32x8d```: Use resnext101_32x8d as image feature extractor.

*Note:* ```resnet101``` *and* ```resnext101_32x8d``` *image feature extractors are only available for* ```ff_bce``` *and* ```lstm```.

Training can be run as in the following example command:
```bash
$ python train.py --save_dir ../checkpoints --resume --seed SEED --dataset DATASET \
--image_model IMAGE_MODEL --model_name MODEL_NAME --use_json_config
```
where DATASET is a dataset name (e.g. `voc`), IMAGE_MODEL and MODEL_NAME are among the models listed above (e.g. `resnet50` and `ff_bce`) and SEED is the value of a random seed (e.g. `1235`).

Check training progress with Tensorboard from ```../checkpoints```:
```bash
$ tensorboard --logdir='.' --port=6006
```

### Evaluation

*Note: all python calls below must be run from* `./src`.

Calculate evaluation metrics as in the following example command:
```bash
$ python eval.py --eval_split test --models_path PATH --dataset DATASET --batch_size 100
```
where DATASET is a dataset name (e.g. `voc`) and PATH is the path to the saved models folder.

### Pre-trained models

We are releasing ```ff_bce``` and ```lstm``` pre-trained models (single seed) for all image backcbones. Please follow the links below:

|| VOC | COCO | NUSWIDE | ADE20k | RECIPE1M |
| :------------- | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| **resnet50**      | <p>[ff_bce](https://dl.fbaipublicfiles.com/image-to-set/voc_resnet50_ff_bce_1235.ckpt) <br>[lstm](https://dl.fbaipublicfiles.com/image-to-set/voc_resnet50_lstm_1235.ckpt)</p> | <p>[ff_bce](https://dl.fbaipublicfiles.com/image-to-set/coco_resnet50_ff_bce_1235.ckpt) <br>[lstm](https://dl.fbaipublicfiles.com/image-to-set/coco_resnet50_lstm_1235.ckpt)</p> | <p>[ff_bce](https://dl.fbaipublicfiles.com/image-to-set/nuswide_resnet50_ff_bce_1235.ckpt) <br>[lstm](https://dl.fbaipublicfiles.com/image-to-set/nuswide_resnet50_lstm_1235.ckpt)</p> | <p>[ff_bce](https://dl.fbaipublicfiles.com/image-to-set/ade20k_resnet50_ff_bce_1235.ckpt) <br>[lstm](https://dl.fbaipublicfiles.com/image-to-set/ade20k_resnet50_lstm_1235.ckpt)</p> | <p>[ff_bce](https://dl.fbaipublicfiles.com/image-to-set/recipe1m_resnet50_ff_bce_1235.ckpt) <br>[lstm](https://dl.fbaipublicfiles.com/image-to-set/recipe1m_resnet50_lstm_1235.ckpt)</p> |
| **resnet101**      | <p>[ff_bce](https://dl.fbaipublicfiles.com/image-to-set/voc_resnet101_ff_bce_1235.ckpt) <br>[lstm](https://dl.fbaipublicfiles.com/image-to-set/voc_resnet101_lstm_1235.ckpt)</p> | <p>[ff_bce](https://dl.fbaipublicfiles.com/image-to-set/coco_resnet101_ff_bce_1235.ckpt) <br>[lstm](https://dl.fbaipublicfiles.com/image-to-set/coco_resnet101_lstm_1235.ckpt)</p> | <p>[ff_bce](https://dl.fbaipublicfiles.com/image-to-set/nuswide_resnet101_ff_bce_1235.ckpt) <br>[lstm](https://dl.fbaipublicfiles.com/image-to-set/nuswide_resnet101_lstm_1235.ckpt)</p> | <p>[ff_bce](https://dl.fbaipublicfiles.com/image-to-set/ade20k_resnet101_ff_bce_1235.ckpt) <br>[lstm](https://dl.fbaipublicfiles.com/image-to-set/ade20k_resnet101_lstm_1235.ckpt)</p> | <p>[ff_bce](https://dl.fbaipublicfiles.com/image-to-set/recipe1m_resnet101_ff_bce_1235.ckpt) <br>[lstm](https://dl.fbaipublicfiles.com/image-to-set/recipe1m_resnet101_lstm_1235.ckpt)</p> |
| **resnext101_32x8d**      | <p>[ff_bce](https://dl.fbaipublicfiles.com/image-to-set/voc_resnext101_32x8d_ff_bce_1235.ckpt) <br>[lstm](https://dl.fbaipublicfiles.com/image-to-set/voc_resnext101_32x8d_lstm_1235.ckpt)</p> | <p>[ff_bce](https://dl.fbaipublicfiles.com/image-to-set/coco_resnext101_32x8d_ff_bce_1235.ckpt) <br>[lstm](https://dl.fbaipublicfiles.com/image-to-set/coco_resnext101_32x8d_lstm_1235.ckpt)</p> | <p>[ff_bce](https://dl.fbaipublicfiles.com/image-to-set/nuswide_resnext101_32x8d_ff_bce_1235.ckpt) <br>[lstm](https://dl.fbaipublicfiles.com/image-to-set/nuswide_resnext101_32x8d_lstm_1235.ckpt)</p> | <p>[ff_bce](https://dl.fbaipublicfiles.com/image-to-set/ade20k_resnext101_32x8d_ff_bce_1235.ckpt) <br>[lstm](https://dl.fbaipublicfiles.com/image-to-set/ade20k_resnext101_32x8d_lstm_1235.ckpt)</p> | <p>[ff_bce](https://dl.fbaipublicfiles.com/image-to-set/recipe1m_resnext101_32x8d_ff_bce_1235.ckpt) <br>[lstm](https://dl.fbaipublicfiles.com/image-to-set/recipe1m_resnext101_32x8d_lstm_1235.ckpt)</p>|

### License

image-to-set is released under MIT license, see [LICENSE](LICENSE.md) for details.
