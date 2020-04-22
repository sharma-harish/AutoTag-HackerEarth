set TRAIN_CSV=F:\Workspace\AutoTag-HE\input\train.csv
set TEST_CSV=F:\Workspace\AutoTag-HE\input\test.csv
set KFOLD_CSV=F:\Workspace\AutoTag-HE\input\train_kfolds.csv

set TRAIN_PATH=F:\Workspace\AutoTag-HE\input\Train Images\
set TEST_PATH=F:\Workspace\AutoTag-HE\input\Test Images\

set IMG_HT=128
set IMG_WD=128

set EPOCHS=50
set TRAIN_BAT_SIZE=64
set TEST_BAT_SIZE=1

set MODEL_MEAN=(0.485, 0.456, 0.406)
set MODEL_STD=(0.229, 0.224, 0.225)
set BASE_MODEL=resnet34

set TRAIN_FOLDS=(0, 1, 2, 3, 4, 5, 6, 9, 8)
set VALID_FOLDS=(7,)

REM REM python -m create_folds
REM python -m train
REM python -m test

REM set TRAIN_FOLDS=(0, 1, 2, 3, 4, 5, 8, 7, 9)
REM set VALID_FOLDS=(6,)

REM python -m train
REM python -m test

REM set TRAIN_FOLDS=(0, 1, 2, 3, 4, 6, 7, 8, 9)
REM set VALID_FOLDS=(5,)

REM python -m train
python -m test

REM python -m merge_CV