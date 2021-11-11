# Imagenet Evaluation
Below you can find results for imagenet evaluation, obtained by running `main.py` in 
this part of repository.

### Results

#### Notice
Please bear in mind that the models in this repository, produce the same outputs as the
models exported from `google/automl` repository (assuming the same input and machine).

The Imagenet Validation Accuracy is expected to fluctuate a bit when
rewriting the models between API's or frameworks.

#### Table
| Variant | Image size | Official Top 1 | Official Top 5  | This repo Top 1 | This repo Top 5 | Top 1 diff | Top 5 diff |
| ------- | ---------- | -------------- |  -------------- |---------------  | --------------- | ---------- | ---------- |
|  B0     |    224     |      78.7      |        94.4     |       78.7      |       94.3      |     0.0    |     0.0    |
|  B1     |    240     |      79.8      |        94.9     |       79.8      |       95.0      |     0.0    |     0.0    |
|  B2     |    260     |      80.5      |        95.1     |       80.5      |       95.1      |     0.0    |     0.0    |
|  B3     |    300     |      82.0      |        95.8     |       82.0      |       95.8      |     0.0    |     0.0    |
|  S      |    384     |      83.8      |        96.7     |       83.9      |       96.7      |     0.0    |     0.0    |
|  M      |    480     |      85.3      |        97.3     |       85.3      |       97.4      |     0.0    |     0.0    |
|  L      |    480     |      85.7      |        97.5     |       85.7      |       97.5      |     0.0    |     0.0    |

#### Why the differences
There are slight differences in measured accuracy.   
One can speculate why. Although not large, the difference might come from:
* The API used: Official uses `TPUEstimator`, I use `tf.keras.Model`
* Hardware used: Official uses TPU, I use GPU.
* Precision: Official runs in `bfloat16`, I use `float32`.

### To reproduce my eval:
The `external/` directory contains code from original repository.

. Download Imagenet and use [imagenet_to_gcs](https://github.com/tensorflow/tpu/blob/acb331c8878ce5a4124d4d7687df5fe0fadcd43b/tools/datasets/imagenet_to_gcs.py) 
script to obtain tfrecords:
```
python imagenet_to_gcs.py \
    --raw_data_dir=imagenet_home/ \
    --local_scratch_dir=my_tfrecords \
    --nogcs_upload
```
        
2. To eval this repo models, run  (for example):
```
python main.py  \
    --variant b0 \
    --data_dir /path/to/tfrecords/validation \
    --weights /path/to/weights \  # Or weight argument
```   

Change parameters accordingly, as in the table above.
