# Self-Supervised Learning-Based Framework for Speckle Reduction of Optical Coherence Tomography Images Using Frame Interpolation


## Introduction
This project is the implement of [Self-Supervised Learning-Based Framework for Speckle Reduction of Optical Coherence Tomography Images Using Frame Interpolation](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202500001). 

In this repo, I add an extra convex upsampling to replace previous SAFMN upsampling module to achieve a more consistent training progress. To further enhance the training process, I upate the loss and training pipeline.

### Installation

```
git clone https://github.com/JayJiang99/FIDenoise.git
cd FIDenoise
pip install -r requirements.txt
```
### Dataset

[PKU37](https://tianchi.aliyun.com/dataset/133217)
[OCT-R1,OCT-R2](https://tianchi.aliyun.com/dataset/161472)
[DUKE](https://people.duke.edu/~sf59/fang_tmi_2013.htm)
[DIOME](https://opara.zih.tu-dresden.de/xmlui/handle/123456789/6047)
[IOVS](https://www.duke.edu/~sf59/Chiu_IOVS_2011_dataset.htm)



### Preparing Dataset
Update the dataset process in dataset.py, self.data_root = 'The DIR of Dataset'


### Inference
```
python inference_denoise.py 
```

### Training
```
bash train.sh
```



## Checklist

- [ ] Release the upsampling training code
- [ ] Release the evaluation code
- [ ] Release the dataset preparation code
- [ ] Update the doc for the new training pipeline




## Citation
If you think this project is helpful, please feel free to leave a star or cite our paper:
```

@article{jiang2025FID,
  title = {Self-Supervised Learning-Based Framework for Speckle Reduction of Optical Coherence Tomography Images Using Frame Interpolation},
  author = {Jiang, Zhiyi and Hao, Yifeng and Dai, Jing and Kwok, Ka-Wai},
  journaltitle = {Advanced Intelligent Systems},
  pages = {2500001},
  issn = {2640-4567},
  year = {2025}
}


```


## Acknowledgement
Our code is based on the implementation of [ScCov](https://github.com/cheng-haha/ScConv.git), [RIFE](https://github.com/hzwer/ECCV2022-RIFE.git) and [SAFMN](https://github.com/sunny2109/SAFMN.git). Thanks to their excellent work and repository.
