# vaik-classification-tflite-experiment

Create json file by classification model. Calc ACC.

------

## Install

```shell
pip install -r requirements.txt
```

### Docker Install

```shell
sudo apt-get update && sudo apt-get upgrade
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```


### armv7l(raspberry pi 4b) without coral

```shell
sudo docker build -t raspberry4b_experiment -f ./Dockerfile.raspberrypib4 .
sudo docker run --name raspberry4b_experiment_container \
           --rm \
           -v ~/.vaik-mnist-classification-dataset/:/workspace/vaik-mnist-classification-dataset \
           -v ~/output_tflite_model:/workspace/output_tflite_model \
           -v $(pwd):/workspace/source \
           -it raspberry4b_experiment /bin/bash
```

### armv7l(raspberry pi 4b) with coral

```shell
sudo docker build -t raspberry4b_experiment -f ./Dockerfile.raspberrypib4 .
sudo docker run --name raspberry4b_experiment_container \
           --rm \
           --privileged \
           -v ~/.vaik-mnist-classification-dataset:/workspace/vaik-mnist-classification-dataset \
           -v ~/output_tflite_model:/workspace/output_tflite_model \
           -v $(pwd):/workspace/source \
           -v /dev/bus/usb:/dev/bus/usb \
           -it raspberry4b_experiment /bin/bash
```

---------

## Usage

### Create json file

```shell
python inference.py --input_saved_model_dir_path '~/output_tflite_model/mnist_mobile_net_v2.tflite' \
                --input_classes_path '~/.vaik-mnist-classification-dataset/classes.txt' \
                --input_image_dir_path '~/.vaik-mnist-classification-dataset/valid' \
                --output_json_dir_path '~/.vaik-mnist-classification-dataset/valid_inference'
```

- input_image_dir_path
  - example

```shell
.
├── eight
│   ├── valid_000000024.jpg
│   ├── valid_000000034.jpg
・・・
│   └── valid_000001976.jpg
├── five
│   ├── valid_000000016.jpg
・・・
```

#### Output
- output_json_dir_path
  - example

```json
{
    "answer": "one",
    "image_path": "/home/kentaro/.vaik-mnist-classification-dataset/valid/one/valid_000000000.jpg",
    "label": [
        "one",
        "seven",
        "four",
        "eight",
        "six",
        "nine",
        "three",
        "zero",
        "five",
        "two"
    ],
    "score": [
        0.9999998807907104,
        7.519184208604202e-08,
        4.9287844916534596e-08,
        1.9263076467268547e-08,
        1.518927739141418e-08,
        2.0775094977665276e-09,
        6.83717971128317e-10,
        1.3445242141862934e-10,
        4.8028431925972725e-11,
        1.8028399953462504e-11
    ]
}
```
-----

### Calc ACC

```shell
python calc_acc.py --input_json_dir_path '~/.vaik-mnist-classification-dataset/valid_inference' \
                --input_classes_path '~/.vaik-mnist-classification-dataset/classes.txt'
```

#### Output

``` text
              precision    recall  f1-score   support

        zero     1.0000    0.9851    0.9925       201
         one     0.9957    1.0000    0.9979       234
         two     0.9783    0.9890    0.9836       182
       three     1.0000    0.9913    0.9956       230
        four     1.0000    0.9946    0.9973       185
        five     0.9829    0.9829    0.9829       175
         six     0.9831    0.9777    0.9804       179
       seven     1.0000    0.9958    0.9979       240
       eight     1.0000    0.9784    0.9891       185
        nine     0.9545    1.0000    0.9767       189

    accuracy                         0.9900      2000
   macro avg     0.9895    0.9895    0.9894      2000
weighted avg     0.9902    0.9900    0.9900      2000
```