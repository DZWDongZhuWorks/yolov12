# satellite EXP 1 @ ada-6000

    基於1600進行路面模型訓練

## dataset preparation

    # 1. consolidate and convert
    python /home/ubuntu/Desktop/Rontgen/dataset_utils/labelme2yolo12seg.py \
    --input /home/ubuntu/Desktop/Rontgen/dataset/satellite_road/ \
    --output /home/ubuntu/Desktop/Rontgen/dataset/satellite_road/yolodataset \
    --names /home/ubuntu/Desktop/Rontgen/yolov12/data/satellite_road-label-names.txt \
    --val-ratio 0.2 --test-ratio 0.0 --copy

    2. offline augmentation (already done)
    python /home/ubuntu/Desktop/Rontgen/dataset_utils/seg_augmentation.py \
    /home/ubuntu/Desktop/Rontgen/dataset/satellite_road/yolodataset/images/train \
    /home/ubuntu/Desktop/Rontgen/dataset/satellite_road/yolodataset/labels/train \
    /home/ubuntu/Desktop/Rontgen/dataset/satellite_road/yolodataset/augmented \
    /home/ubuntu/Desktop/Rontgen/dataset_utils/hyp.yaml \
    --new_image 2

## train @ conda env: yolov12

    ``` 解決 遇到 device 序號問題
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES=1                       # 只暴露實體 GPU#1
    # export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # Caused GPU driver deadlock on Ada 6000
    ```


    nohup python /home/ubuntu/Desktop/Rontgen/yolov12/train.py \
    --model /home/ubuntu/Desktop/Rontgen/yolov12/yolov12x-seg.pt \
    --pretrained "" \
    --data /home/ubuntu/Desktop/Rontgen/dataset/satellite_road/yolodataset/dataset.yaml \
    --project satellite_road --name e1 \
    --imgsz 1600 --batch 1 \
    -- \
    device=0 warmup_epochs=8 \
    patience=300 close_mosaic=60  \
    hsv_h=0.000 hsv_s=0.7 hsv_v=0.4 \
    degrees=10 translate=0.1 shear=0.0 perspective=0.0 \
    fliplr=0.0 flipud=0.5 \
    copy_paste=0.0 \
    cos_lr=True optimizer=SGD momentum=0.937 weight_decay=0.0005 \
    &> logs/satellite_road_e1.txt &
