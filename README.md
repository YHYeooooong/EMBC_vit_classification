# EMBC_vit_classification
2023.03 유방암 분류 성능 비교

### Intro
CNN은 computer vision 분야에서 전통적으로 막강한 성능을 보여주는 딥러닝 모델 중 하나입니다. 그러나 최근 NLP 분야의 Transformer 구조를 기초로 하는 모델과 MLP-Mixer 모델이 새롭게 등장하며 CNN과 비슷하거나 더 월등한 성능을 보여주었습니다. 이러한 모델들의 등장은 Computer vision 모델의 성능을 향상 시키는데 많은 영향을 주었으며, transformer, MLP-Mixer기반의 모델이 classification, Object detection, Segmentation 과제에서 좋은 성능을 기록함을 확인할 수 있었습니다. 하지만, Medical image 분야에서, 특히 유방암 분류과제에서의 Transformer 및 MLP-Mixer 모델의 성능에 대한 연구는 충분히 이루어지지 않았고, 이를 해결하기 위해 5개의 Transformer 및 MLP-Mixer 기반의 모델과 2개의 CNN 모델 사이의 성능을 간단하게 비교해보고자 하였습니다. 

### 목표
데이터셋 CBIS-DDSM 을 활용하여 5개의 Transformer 모델과, 2개의 CNN 모델 사이의 분류 성능을 Acc 기반으로 비교하는 것

### 데이터셋
CBIS-DDSM : public 유방암 데이터셋
Class : Benign, Malgnant

