#trianing 코드에 How to train vision transformer 논문 내용을 적용시킨것
#그냥 base 21k 모델 사용
#auc 점수를 계산, tensorboard 출력하는 목표
#lr 변화 밑 보드 결과 준비물 저장 확인용


import timm
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import time
import os
import copy


from torch.utils.tensorboard import SummaryWriter
from tensorboard.resnet_tensorboard import write_tensorboard
'''
auc 를 구현하기 위해서는 예측한 클래스의 점수가 필요함
y_hat, y_test, probability는 마지막 에폭의 validation phase에서만 저장하면 됨
1. 마지막 에폭 알아채기
2. y_hat, y_test. probability 저장하기

'''

torch.cuda.empty_cache()
Epoch = 300
Learning_rate = 0.0005
Mommentum = 0.9
Weight_decay = 0.0005
num_classes=2
batch_size = 30
file_name = '20220407_Resnet50_e'+str(Epoch)+'_opti_SGD_lr'+str(Learning_rate)+'_bat_'+str(batch_size)+'_+aaug_clip_1'
device_num = 1

device = torch.device("cuda:"+str(device_num) if torch.cuda.is_available() else "cpu")
CCpu = torch.device('cpu')

model = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)

## dino 모델도 timm에서 제공하는 것을 확인했음 timm으로 불러올 수 있을듯! num_clasees =2 를 쓰면 자동으로 마지막 레이어도 붙여주는 것 확인
# vit base 8x8 의 경우 최대 batch size 는 6임
model = model.to(device) #위에서 선언한 모델을 gpu로 올려주기
optimizer_ft = optim.SGD(model.parameters(), lr=Learning_rate, momentum=Mommentum, weight_decay=Weight_decay) # 위에서 설정한값들을 가지고 옵티마이저 설정

total_path = "../models/"+file_name+'/' # 모델 저장 경로 설정
if not os.path.isdir("../models/"+file_name) :
    os.mkdir("../models/"+file_name) # 이거 디렉토리가 겹치는 이슈 해결함
writer = SummaryWriter(total_path+'/tensorboard')




# 검증을 위한 일반화
data_transforms = {
    'train': transforms.Compose([
     #훈련 이미지 리사이즈, 이런 transforms를 넣어도 2배가 되거나 하지는 않음
        transforms.ToTensor() # 텐서로 바꾸기
        
    ]),
    'train_vertiflip': transforms.Compose([
    #훈련 이미지 리사이즈
    transforms.RandomVerticalFlip(p=1),
    transforms.ToTensor() # 텐서로 바꾸기
    ]),
    'train_holiflip': transforms.Compose([
    #훈련 이미지 리사이즈
    transforms.RandomHorizontalFlip(p=1),
    transforms.ToTensor() # 텐서로 바꾸기
    ]),
    'train_color_jit': transforms.Compose([
    #훈련 이미지 리사이즈
    transforms.ColorJitter(brightness=5, hue=.3), #color jitter 파이토치 홈페이지의 예제 값을 사용했음
    transforms.ToTensor() # 텐서로 바꾸기
    ]),
    
    'train_rancropresize': transforms.Compose([
    #훈련 이미지 리사이즈
    transforms.RandomResizedCrop(size=(224,224)), #randcropresize 디폴트 값을 사용한다
    transforms.ToTensor() # 텐서로 바꾸기
    ]),

    'val': transforms.Compose([
         # 테스트 이미지 리사이즈
        transforms.ToTensor() # 텐서로 바꾸기
    ]),
}

image_datasets = {}

data_dir ='../jpg_224' # 이미지를 가져오는 장소



train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), 
                                        data_transforms['train']) # train 이미지 가져와서  resize만 시키기

train_verti = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                        data_transforms['train_vertiflip']) # train 이미지 가져와서 Verticalflip만 시키기

train_hori = datasets.ImageFolder(os.path.join(data_dir, 'train'), 
                                        data_transforms['train_holiflip']) # train 이미지 가져와서 Horizontalflip만 시키기

train_Cjitter = datasets.ImageFolder(os.path.join(data_dir, 'train'), 
                                        data_transforms['train_color_jit']) # train 이미지 가져와서 color jitter만 시키기

train_ranrecrop = datasets.ImageFolder(os.path.join(data_dir, 'train'), 
                                        data_transforms['train_rancropresize']) # train 이미지 가져와서 random resize crop만 시키기                                        
total_train = torch.utils.data.ConcatDataset([train_data, train_verti, train_hori, train_Cjitter, train_ranrecrop]) # augmented train 이미지를 가져와서 하나의 dataset으로 묶기

val_data = datasets.ImageFolder(os.path.join(data_dir, 'val'), 
                                        data_transforms['val']) # val 이미지 가져와서 이미지 resize만 시키기

image_datasets['train'] = total_train # 합친 데이터셋을 train 딕셔너리에 넣음
image_datasets['val'] = val_data # val 데이터셋을 val 딕셔너리에 넣음



print(image_datasets.keys())
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, ## batch size 만들어진 data set을 로더로 넣음
                                            shuffle=True, num_workers=4) # 데이터 셔플
                                            for x in ['train', 'val']} 
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']} # 데이터셋 사이즈 --> 데이터 갯수가 원래 데이터 갯수가 맞는것 확인
#class_names = image_datasets['train'].classes # 클래스 이름  --> 둘다 맞는 것 확인
print(dataset_sizes)




result = {}
train_acc = []
train_loss = []
val_acc = []
val_loss = []

def train_model(model, criterion, optimizer, path, writer,  num_epochs=25, result=result):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    max_norm = 1

    best_result = {}
    
    material = {}
    material['y_hat'] = []
    material['y_test'] = []
    material['y_probab'] = []

    board_result = {}
    board_result['recall'] = []
    board_result['precision'] = []
    board_result['AUC'] = []
    board_result['lr'] = []

    T_y_hat = []
    T_y_test = []
    T_y_probab = []

    T_recall = []
    T_precision = []
    T_lr = []
    T_auc = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)


        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0
            total_num=0

            probab = []
            y_hat = []
            y_test = []

            # 데이터를 반복
            for inputs, labels in dataloaders[phase]: # phase 에 따라 dataloaders['train'], dataloaders['val'] 이 사용됨
                ## 배치사이즈 횟수 몇개인지 출력하는 코드 짜야함 train에 132번 들어감 (batch size 10 기준, 훈련데이터 1318개)
                ## 테스트 데이터 갯수 38개의 배치가 들어가고 전체 데이터 갯수는 378개

                inputs = inputs.to(device)
                labels = labels.to(device) #gpu 로 옮겨주기

                # 매개변수 경사도를 0으로 설정
                optimizer.zero_grad()

                # 순전파
                # 학습 시에만 연산 기록을 추적
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                        optimizer.step() 


                # 통계
                running_loss += loss.item() * inputs.size(0) # 배치당 로스
                running_corrects += torch.sum(preds == labels.data) # 배치당 맞은 갯수 더하기
                total_num += len(labels.data) # 각 배치사이즈당 이미지 수 더해서 한 에폭의 훈련페이즈당 총 이미지 갯수 구하기

                ## probability 출력을 위한 테스트 코드
                #cross entropy loss는 (파이토치에서는) log_softmax  + loss 이다.
                #그래서 outputs에 따로 softmax를 취해줄 것이다.

                
                one_soft= nn.Softmax(dim = 1) 
                soft_output = one_soft(outputs)
                soft_outputs = soft_output.to(CCpu).tolist() # cpu로 옮기고 list로 바꿈
                probab.extend(soft_outputs) # 저장할 리스트에 extend로 입력
                    ## probability 출력을 위한 테스트 코드

                y_hat.extend(preds.to(CCpu).tolist()) # 예측 값을 저장하는 리스트
                y_test.extend(labels.data.to(CCpu).tolist()) # 정답값을 저장하는 리스트 # 각 배치마다의 정답 및 예측 값 저장
            

            epoch_loss = running_loss / dataset_sizes[phase] # 에폭당 평균 로스
            epoch_acc = running_corrects.double() / dataset_sizes[phase] # 에폭당 평균 정확도

            # print(y_test)
            # print(y_hat)

            correct_num = running_corrects.to(CCpu).tolist() # 에폭이 끝나면 맞은 갯수 정하기
            wrong_num = total_num - correct_num # 에폭이 끝나면 틀린갯수 정하기


            # print('correct num ',correct_num)
            # print('total num ',total_num) # 맞는거 확인했음 

            
            epoch_loss = epoch_loss#.to(CCpu) 
            epoch_acc = epoch_acc.to(CCpu) # acccuracy 가 tensor로 반환이 되어서 이것을 cpu로 넘파이 형태로 복사해줌 근데 이거 저장이 이상하게 됨 
            epoch_acc = epoch_acc.item() # 그래서 텐서의 아이템을 가져오는 함수를 사용해서 옮겨봤음 --> 해결 완료!
            ## 이부분에 훈련, 테스트 페이즈 마다 정확도랑 로스 저장하는 변수 만들어서  csv 에 저장하는 코드 짜야함
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            # tensorboard 기록 부분
            writer, E_y_hat, E_y_test, E_y_probab, E_recall, E_precision, E_lr, E_auc = write_tensorboard(writer, phase, epoch, epoch_acc, 
            epoch_loss, y_hat, y_test, probab, optimizer)
            # writer를 받아서 보드에 기록
            # phase를 받아서 phase + acc, phase + loss를 기록
            # phase를 받아서 val일 때, precison, recall, 
            # 한 에폭당 훈련, 테스트 페이즈가 끝날 때 마다 기록함

            # 모델을 깊은 복사(deep copy)함
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                best_yhat = E_y_hat
                best_ytest = E_y_test
                best_y_probab = E_y_probab
                best_recall = E_recall
                best_precision = E_precision
                best_lr = E_lr
                best_AUC =  E_auc
                best_epoch = epoch

            if phase == 'train' :
                train_acc.append(epoch_acc) # 학습 phase 때의 정확도 기록
                train_loss.append(epoch_loss) # 학습 phase 때의 손실 기록
            
            elif phase == 'val' :
                val_acc.append(epoch_acc)
                val_loss.append(epoch_loss)

                T_y_hat.append(E_y_hat)
                T_y_test.append(E_y_test)
                T_y_probab.append(E_y_probab)

                T_recall.append(E_recall)
                T_precision.append(E_precision)
                T_lr.append(E_lr)
                T_auc.append(E_auc)

            if (phase == 'val') and (epoch == num_epochs - 1) :
                val_y_hat = y_hat
                val_y_test = y_test
                val_probab = probab 
                check_cor = 0
                check_wrong = 0

                ## start correct num, wrong num, 확인용 코드
                # 여기에서 correnct num 이랑, y_tesy==y_hat 확이나긴
                # for ind_num in range(len(val_y_test)) :
                #     if y_hat[ind_num] == y_test[ind_num] :
                #         check_cor +=1
                #     else : check_wrong +=1
                
                # print('correct num ', correct_num)
                # print('check correct num ',check_cor)
                # print(check_cor == correct_num)
                # print(check_wrong == (total_num-correct_num))
                ## end correct num, wrong num, 확인용 코드


        print()
    writer.close()
    result['train_acc'] = train_acc # 저장된 acc와 loss 를 다 받아서 저장하기
    result['train_loss'] = train_loss
    result['val_acc'] = val_acc
    result['val_loss'] = val_loss
    #print(result)
    result = pd.DataFrame(result)
    result.to_csv(path + file_name +'_acc_loss.csv') # CSV로 저장하기 잘 되는 것 확인

    con_csv = {}
    con_csv['val_y_hat'] = val_y_hat
    con_csv['val_y_test'] = val_y_test
    con_csv['correct'] = correct_num
    con_csv['wrong'] = wrong_num
    con_csv['probabilities'] = probab

    con_csv = pd.DataFrame(con_csv)
    con_csv.to_csv(path + file_name +'_hat_test.csv') #  csv로 저장이 잘 되는것 확인

    material['y_hat'] = T_y_hat
    material['y_test'] = T_y_test
    material['y_probab'] = T_y_probab

    board_result['recall'] = T_recall
    board_result['precision'] = T_precision
    board_result['AUC'] = T_auc
    board_result['lr'] = T_lr
    
    material = pd.DataFrame(material)
    board_result =pd.DataFrame(board_result)
    material.to_csv(path+file_name+'_board_materials.csv')
    board_result.to_csv(path+file_name+'_board_result.csv') # 텐서보드의 계산결과와 그 준비물을 저장

    #최고 점수
    best_result['Best_acc'] = best_acc 
    best_result['Best_loss'] = best_loss
    best_result['Best_yhat'] = best_yhat 
    best_result['Best_ytest'] = best_ytest 
    best_result['Best_y_probab'] = best_y_probab 
    best_result['Best_recall'] = best_recall 
    best_result['Best_precision'] = best_precision 
    best_result['Best_lr'] = best_lr
    best_result['Best_AUC'] = best_AUC 
    best_result['Best_epoch'] = best_epoch

    best_result = pd.DataFrame(best_result)
    best_result.to_csv(path+file_name+'_best_result.csv')
    # 최고점수 
    
    time_elapsed = time.time() - since 
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 가장 마지막 epoch 의 가중치를 불러옴
    model.state_dict()  ## 베스트 모델 가중치 가져오는거 바꾸기
    return best_model_wts, model, optimizer

    ## 모델 저장하는 코드 넣어야함




criterion = nn.CrossEntropyLoss() #loss 설정 

# 모든 매개변수들이 최적화되었는지 관찰




best_model_wts, model_ft, optimizer_ft = train_model(model, criterion, optimizer_ft, total_path, writer,  num_epochs=Epoch, result=result) # 모델 훈련


torch.save(best_model_wts, total_path + file_name +'best_model.pt') # 최고 모델 저장
torch.save(model_ft, total_path + file_name +'.pt')