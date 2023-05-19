# EMBC_vit_classification
2023.03 유방암 분류 성능 비교

## Intro
CNN은 computer vision 분야에서 전통적으로 막강한 성능을 보여주는 딥러닝 모델 중 하나입니다. 그러나 최근 NLP 분야의 Transformer 구조를 기초로 하는 모델과 MLP-Mixer 모델이 새롭게 등장하며 CNN과 비슷하거나 더 월등한 성능을 보여주었습니다. 이러한 모델들의 등장은 Computer vision 모델의 성능을 향상 시키는데 많은 영향을 주었으며, transformer, MLP-Mixer기반의 모델이 classification, Object detection, Segmentation 과제에서 좋은 성능을 기록함을 확인할 수 있었습니다. 하지만, Medical image 분야에서, 특히 유방암 분류과제에서의 Transformer 및 MLP-Mixer 모델의 성능에 대한 연구는 충분히 이루어지지 않았고, 이를 해결하기 위해 5개의 Transformer 및 MLP-Mixer 기반의 모델과 2개의 CNN 모델 사이의 성능을 간단하게 비교해보고자 하였습니다. 

## Project
### 목표
데이터셋 CBIS-DDSM 을 활용하여 5개의 Transformer 모델과, 2개의 CNN 모델 사이의 분류 성능을 Acc 기반으로 비교하는 것

### 데이터셋
실험을 위해 CBIS-DDSM 유방암 이미지 데이터셋을 사용하였다. [CBIS-DDSM](https://www.google.com/search?q=CBIS-DDSM&rlz=1C1PNBB_koKR948KR948&oq=CBIS-DDSM&aqs=chrome..69i57j0i512l4j69i65j69i60l2.2736j0j7&sourceid=chrome&ie=UTF-8)
해당 데이터셋은 전체 mammogram 이미지, 각 이미지 데이터의 유방암 mask, 각 이미지 데이터의 ROI 이미지로 구성되어있다. 또한, 훈련데이터와 테스트 데이터를 따로 제공하며, 각 이미지의 위치와 class에 대한 정보는 csv 파일로 정리되어있다. 또한 이미지들은 DICOM 형태로 제공되기 때문에, 추가적인 전처리가 필요하다.

##### Preprocessing
1. Pydicom 라이브러리를 사용하여 DICOM --> jpg 형태로 전환 [dicom_convert_total.py](https://github.com/YHYeooooong/EMBC_vit_classification/blob/main/preprocessing/dicom_convert_total.py)
2. 224x224 사이즈로 일괄적으로 resize [resize_jpg.py](https://github.com/YHYeooooong/EMBC_vit_classification/blob/main/preprocessing/resize_jpg.py)
3. train, test set에 따라 해당하는 폴더로 이동

### 실험
VIT 모델 3개, MLP-Mixer 모델 2개, CNN 모델 2개를 선정하여 유방암 ROI 에 대해서 Benign과 Malignant를 분류하도록 훈련시키고, 분류성능을 비교해 보았다.

##### 사용한 모델 및 사전학습 설정

| Model name  | pretrained setting |
| ------------- | ------------- |
| BEiT-base  | ImageNet21k  |
| CvT-21  | ImageNet21k  |
| CvT-W24  | ImageNet21k  |
| MLP-Mixer b16 | ImageNet21k  |
| MLP-Mixer L16  | ImageNet21k  |
| DenseNet121 | ImageNet  |
| ResNet50 | ImageNet  |

##### 분류 성능

![결과 표](https://github.com/YHYeooooong/EMBC_vit_classification/assets/43724177/ee69d6aa-b559-4e60-b970-91a3ec02d410)

정확도를 기준으로 ViT류 모델 중 CvT-W24 가 75.4%로 가장 높은 분류성능을 달성하였으며, CNN 모델보다 더 높은 분류정확도와 loss 값을 가짐을 알 수 있습니다. 앞으로의 ViT와 MLP-Mixer류의 모델에 대한 연구가 계속해서 진행될 것을 감안한다면, ViT 및 MLP-Mixer류의 모델의 유방암 분류 과제에 대한 성능이 CNN 모델을 뛰어넘을 수 있는 잠재력이 크다고 볼 수 있습니다. 
