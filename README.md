# Guidance to Use the Codes of MVS-Net
## 1.Recommended Environment: 
```
Python 3.7
Pytorch 1.10.0
torchvision 0.10.0
numpy 1.21.6
scipy 1.7.3
```
## 2.Data preparation: 
Downloading training and testing datasets and move them into ./datasets/, which can be found in this [Google Drive](https://drive.google.com/file/d/1bQOy4YoJAYcJ9cMgXy_yn0IbxiaGK9W3/view?usp=sharing).

## 3.Pretrained model:  
You should download the pretrained model from [Google Drive](https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV?usp=sharing), and then put it in the './pretrained_pth' folder for initialization. 

## 4.Well trained model:
You could download the trained model from [Google Drive](https://drive.google.com/file/d/1t98vV5ZjLCwNl6YLw8zGx-pSU34UX7Ia/view?usp=sharing) and put the model in directory './checkpoint'.

## 5.Training:
The training results can be found in ./model_pth/. You can try the following code for training:
```
cd code
python Train.py
```

## 6.Testing:
The testing results can be found in a new fold ./result_map/. You can try the following code for testing:
```
cd code
python Test.py
```

## 7 Evaluating your trained model:
The evaluating results can be found in a new fold ./results/. You can try the following code for evaluating:
```
cd code
python Eval.py
```