## Introduction
YOLO-v3 implemention from "[YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)". <br>
<p align="center"><img width="40%" src="result/res_man.jpeg" /></p>

## Tutorial
Get tutorial series in [HonePage](https://ne7ermore.github.io/post/yolo-v3/) or [ZhihuPage](https://zhuanlan.zhihu.com/p/36298401) if know Chinese

## Requirement
* python 3.6
* pytorch 0.4.0
* numpy 1.13.1
* PIL

## Usage

### Options
```
usage: detect.py [-h] [--images IMAGES] [--result RESULT]
                 [--batch_size BATCH_SIZE] [--img_size IMG_SIZE]
                 [--confidence CONFIDENCE] [--nms_thresh NMS_THRESH]
                 [--weights WEIGHTS] [--no_cuda]

YOLO-v3 Detect

optional arguments:
  -h, --help            show this help message and exit
  --images IMAGES
  --result RESULT
  --batch_size BATCH_SIZE
  --img_size IMG_SIZE
  --confidence CONFIDENCE
  --nms_thresh NMS_THRESH
  --weights WEIGHTS
  --no_cuda
```

### Detect
Download [yolo.v3.coco.weights.pt](https://pan.baidu.com/s/1T132ayZiVsWLD3XQCbFKxQ) to $(PROJECT_HOME)/
```
python3 detect.py
```

## Citation
If you find this code useful for your research, please cite:
```
@misc{TaoYOLOv3,
  author = {Ne7ermore Tao},
  title = {yolo-v3},
  publisher = {GitHub},
  year = {2018},
  howpublished = {\url{https://github.com/ne7ermore/yolo-v3}}
}
```

## Contact
Feel free to contact me if there is any question (Tao liaoyuanhuo1987@gmail.com).
