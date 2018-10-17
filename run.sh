#!/bin/bash

# Set paths
INCEPTION_V1_PRETRAINED=pretrained/bvlc_googlenet.caffemodel.pth
CPM_PRETRAINED=pretrained/pose_iter_440000.caffemodel.pth

MARKET1501_DATA_ROOT="/userhome/reid/part_bilinear_reid/data/market1501/raw/Market-1501-v15.09.15"

export MARKET1501_DATA_ROOT=$MARKET1501_DATA_ROOT
export MARKET1501_TRAIN_LIST=$MARKET1501_TRAIN_LIST
export INCEPTION_V1_PRETRAINED=$INCEPTION_V1_PRETRAINED
export CPM_PRETRAINED=$CPM_PRETRAINED

PYTHON='python'
GPU_ID="0,1"

# model
ARCH='inception_v1_cpm'
FEATURES=512
DILATION=2
USE_RELU=false

# data
DATASET='market1501'
HEIGHT=160
WIDTH=80
CROP_HEIGHT=160
CROP_WIDTH=80
BATCH_SIZE=250
USE_CAFFE_SAMPLER=true # [true | false]

# optimizer
OPTIMIZER='sgd_caffe'
LR=0.01
WEIGHT_DECAY=0.0002
EPOCHS=750

LOG_DIR="logs/$DATASET/$EXP_ID"


# Download pretrained network weights
if ! [ -d pretrained ]; then
    mkdir -p pretrained
fi
if ! [ -f "pretrained/bvlc_googlenet.caffemodel.pth" ]; then
    echo "Downloading pretrained inception_v1..."
    wget "https://www.dropbox.com/s/2ljm35ztj6hllcu/bvlc_googlenet.caffemodel.pth?dl=0" -O pretrained/bvlc_googlenet.caffemodel.pth
    echo "Done!"
fi
if ! [ -f "pretrained/pose_iter_440000.caffemodel.pth" ]; then
    echo "Downloading pretrained cpm..."
    wget "https://www.dropbox.com/s/pzb3ow1793yf8dc/pose_iter_440000.caffemodel.pth?dl=0" -O pretrained/pose_iter_440000.caffemodel.pth
    echo "Done!"
fi


# Make log directory
if [ -d $LOG_DIR ]; then
    echo "Same experiment already exists. Change the exp name and retry!"
    exit
else
    mkdir -p $LOG_DIR
    cp $CONFIGURE_PATH "$LOG_DIR/args"
fi

# Parameters
STR_PARAM="-d $DATASET -b $BATCH_SIZE -j 4 -a $ARCH --logs-dir $LOG_DIR --margin 0.2 --features $FEATURES --width $WIDTH --height $HEIGHT --crop-height $CROP_HEIGHT --crop-width $CROP_WIDTH --lr $LR --epochs $EPOCHS --dilation $DILATION --weight-decay $WEIGHT_DECAY"

if [ "$USE_CAFFE_SAMPLER" = true ]; then
    STR_PARAM="$STR_PARAM --caffe-sampler"
fi

cd  /userhome/reid/part_bilinear_reid
# Run!
python train.py $STR_PARAM