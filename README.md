# CycleGAN
Tensorflow implementation of CycleGAN.
* [Original implementation](https://github.com/junyanz/CycleGAN/)
* [Paper](https://arxiv.org/abs/1703.10593)

## Environment
* TensorFlow-gpu 1.3.0
* Python 2.7.12

## Data preparing
First, download a dataset:
```
bash download_dataset.sh monet2photo
```
Second, write the dataset to tfrecords:
```
python build_data.py
```

## Training
Quick train:
```
python main.py
```
Or continue training from a pre-trained model:
```
python main.py --pre_trained 20180117-1030
```

## Results
See training details and images in TensorBoard:
```
tensorboard --logdir checkpoints/${datetime}
```
