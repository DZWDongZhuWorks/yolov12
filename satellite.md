# satellite EXP 1

## convert dataset
    python /home/rtx5090/Documents/Rontgen/dataset_utils/labelme2yolo12seg.py \
    --input /home/rtx5090/Documents/Rontgen/dataset/satellite/109 \
    --output /home/rtx5090/Documents/Rontgen/dataset/satellite/109/yolodataset \
    --names /home/rtx5090/Documents/Rontgen/yolov12/data/satellite-label-names.txt \
    --val-ratio 0.2 --test-ratio 0.0 --copy



## train @ conda env: Rontgen
    nohup python /home/rtx5090/Documents/Rontgen/yolov12/train.py \
    --model /home/rtx5090/Documents/Rontgen/yolov12/yolov12x-seg.pt \
    --data /home/rtx5090/Documents/Rontgen/dataset/satellite/109/YOLODataset2/dataset.yaml \
    --project satellite \
    --name e1 \
    --imgsz 640 \
    --batch 160 \
    &> logs/satellite_e1.txt &
## log
    tail -f logs/satellite_e1.txt

## export 
    python /home/rtx5090/Documents/Rontgen/yolov12/export.py \
    --weights /home/rtx5090/Documents/Rontgen/yolov12/satellite/train_e1/weights/best.pt \
    --include onnx

# satellite EXP 2

## train @ conda env: Rontgen
nohup python /home/rtx5090/Documents/Rontgen/yolov12/train.py \
  --model /home/rtx5090/Documents/Rontgen/yolov12/yolov12x-seg.pt \
  --data /home/rtx5090/Documents/Rontgen/dataset/satellite/109/YOLODataset2/dataset.yaml \
  --project satellite \
  --name e2 \
  --imgsz 640 \
  --batch 160 \
  -- \
  cache=True workers=12 \
  hsv_h=0.000 hsv_s=0.7 hsv_v=0.4 \
  degrees=180 translate=0.1 shear=0.0 perspective=0.0 \
  fliplr=0.5 flipud=0.5 \
  mosaic=1.0 mixup=0.1 copy_paste=0.0 \
  cos_lr=True optimizer=SGD lr0=0.01 momentum=0.937 weight_decay=0.0005 \
  &> logs/satellite_e2.txt &


## log
    tail -f logs/satellite_e2.txt
    tensorboard --logdir /home/rtx5090/Documents/Rontgen/yolov12/satellite --bind_all

# satellite EXP 3

## train @ conda env: Rontgen
nohup python /home/rtx5090/Documents/Rontgen/yolov12/train.py \
  --model /home/rtx5090/Documents/Rontgen/yolov12/yolov12x-seg.pt \
  --data /home/rtx5090/Documents/Rontgen/dataset/satellite/109/YOLODataset2/dataset.yaml \
  --project satellite \
  --name e3 \
  --imgsz 640 \
  --batch 4 \
  --exist_ok \
  -- \
  cache=True workers=12 \
  hsv_h=0.000 hsv_s=0.7 hsv_v=0.4 \
  degrees=180 translate=0.1 shear=0.0 perspective=0.0 \
  fliplr=0.5 flipud=0.5 \
  mosaic=1.0 mixup=0.1 copy_paste=0.0 \
  cos_lr=True optimizer=SGD lr0=0.01 momentum=0.937 weight_decay=0.0005 \
  &> logs/satellite_e3.txt &


## log
    tail -f logs/satellite_e3.txt
    tensorboard --logdir /home/rtx5090/Documents/Rontgen/yolov12/satellite --bind_all

# satellite EXP 4

## train @ conda env: Rontgen
nohup python /home/rtx5090/Documents/Rontgen/yolov12/train.py \
  --model /home/rtx5090/Documents/Rontgen/yolov12/yolov12x-seg.pt \
  --pretrained "" \
  --data /home/rtx5090/Documents/Rontgen/dataset/satellite/109/YOLODataset2/dataset.yaml \
  --project satellite --name e2-b1 \
  --imgsz 1080 --batch 1 \
  -- \
  nbs=1 lr0=0.003 warmup_epochs=8 \
  patience=300 close_mosaic=60  \
  &> logs/satellite_e2_b1.txt &

## log
    tail -f logs/satellite_e2_b1.txt
    tensorboard --logdir /home/rtx5090/Documents/Rontgen/yolov12/satellite --bind_all

# satellite EXP 5

## convert dataset
    python /home/rtx5090/Documents/Rontgen/dataset_utils/labelme2yolo12seg.py \
    --input /home/rtx5090/Documents/Rontgen/dataset/satellite/ \
    --output /home/rtx5090/Documents/Rontgen/dataset/satellite/yolodataset \
    --names /home/rtx5090/Documents/Rontgen/yolov12/data/satellite-label-names.txt \
    --val-ratio 0.2 --test-ratio 0.0 --copy

## train @ conda env: Rontgen
nohup python /home/rtx5090/Documents/Rontgen/yolov12/train.py \
  --model /home/rtx5090/Documents/Rontgen/yolov12/yolov12x-seg.pt \
  --pretrained "" \
  --data /home/rtx5090/Documents/Rontgen/dataset/satellite/yolodataset/dataset.yaml \
  --project satellite --name e5 \
  --imgsz 1080 --batch 1 \
  -- \
  nbs=1 lr0=0.003 warmup_epochs=8 \
  patience=300 close_mosaic=60  \
  &> logs/satellite_e5.txt &

## log
    tail -f logs/satellite_e5.txt
    tensorboard --logdir /home/rtx5090/Documents/Rontgen/yolov12/satellite --bind_all

# satellite EXP 6

## convert dataset
    python /home/rtx5090/Documents/Rontgen/dataset_utils/labelme2yolo12seg.py \
    --input /home/rtx5090/Documents/Rontgen/dataset/satellite/ \
    --output /home/rtx5090/Documents/Rontgen/dataset/satellite/yolodataset \
    --names /home/rtx5090/Documents/Rontgen/yolov12/data/satellite-label-names.txt \
    --val-ratio 0.2 --test-ratio 0.0 --copy

## train @ conda env: Rontgen
nohup python /home/rtx5090/Documents/Rontgen/yolov12/train.py \
  --model /home/rtx5090/Documents/Rontgen/yolov12/yolov12x-seg.pt \
  --pretrained "" \
  --data /home/rtx5090/Documents/Rontgen/dataset/satellite/yolodataset/dataset.yaml \
  --project satellite --name e6_b1 \
  --imgsz 1080 --batch 1 \
  -- \
  nbs=1 lr0=0.003 warmup_epochs=8 \
  patience=300 close_mosaic=60  \
  &> logs/satellite_e6_b1.txt &

# satellite EXP 6a
  exp6a 在 exp6 基礎上加入各種擴增方法

## train @ conda env: Rontgen
nohup python /home/rtx5090/Documents/Rontgen/yolov12/train.py \
  --model /home/rtx5090/Documents/Rontgen/yolov12/yolov12x-seg.pt \
  --pretrained "" \
  --data /home/rtx5090/Documents/Rontgen/dataset/satellite/yolodataset/dataset.yaml \
  --project satellite --name e6_b1_a \
  --imgsz 1080 --batch 1 \
  -- \
  nbs=1 lr0=0.003 warmup_epochs=8 \
  patience=300 close_mosaic=60  \
  hsv_h=0.000 hsv_s=0.7 hsv_v=0.4 \
  degrees=180 translate=0.1 shear=0.0 perspective=0.0 \
  fliplr=0.5 flipud=0.5 \
  mosaic=1.0 mixup=0.1 copy_paste=0.0 \
  cos_lr=True optimizer=SGD lr0=0.01 momentum=0.937 weight_decay=0.0005 \
  &> logs/satellite_e6_b1_a.txt &

## export 
    python /home/rtx5090/Documents/Rontgen/yolov12/export.py \
    --weights /home/rtx5090/Documents/Rontgen/yolov12/satellite/train_e6_b1_a/weights/best.pt \
    --include onnx

# satellite EXP 6-hd @ ada-6000
  將影像提高到 1600 訓練
  遇到 device 序號問題
  ```
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES=1                       # 只暴露實體 GPU#1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ```
## convert dataset
    python /home/ubuntu/Desktop/Rontgen/dataset_utils/labelme2yolo12seg.py \
    --input /home/ubuntu/Desktop/Rontgen/dataset/satellite/ \
    --output /home/ubuntu/Desktop/Rontgen/dataset/satellite/yolodataset \
    --names /home/ubuntu/Desktop/Rontgen/yolov12/data/satellite-label-names.txt \
    --val-ratio 0.2 --test-ratio 0.0 --copy

## train @ conda env: Rontgen
nohup python /home/ubuntu/Desktop/Rontgen/yolov12/train.py \
  --model /home/ubuntu/Desktop/Rontgen/yolov12/yolov12x-seg.pt \
  --data /home/ubuntu/Desktop/Rontgen/dataset/satellite/yolodataset/dataset.yaml \
  --project satellite --name e6_b1_hd \
  --imgsz 1600 --batch 1  \
  -- \
  warmup_epochs=8 \
  patience=300 close_mosaic=60  \
  &> logs/satellite_e6_b1_hd.txt &

# satellite EXP 6-hd-a @ ada-6000
  將影像提高到 1600 訓練並擴增
  遇到 device 序號問題
  ```
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES=1                       # 只暴露實體 GPU#1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ```

## train @ conda env: Rontgen
nohup python /home/ubuntu/Desktop/Rontgen/yolov12/train.py \
  --model /home/ubuntu/Desktop/Rontgen/yolov12/yolov12x-seg.pt \
  --pretrained "" \
  --data /home/ubuntu/Desktop/Rontgen/dataset/satellite/yolodataset/dataset.yaml \
  --project satellite --name e6_b1_hd_a \
  --imgsz 1600 --batch 1 \
  -- \
  device=0 warmup_epochs=8 \
  patience=300 close_mosaic=60  \
  hsv_h=0.000 hsv_s=0.7 hsv_v=0.4 \
  degrees=10 translate=0.1 shear=0.0 perspective=0.0 \
  fliplr=0.0 flipud=0.5 \
  copy_paste=0.0 \
  cos_lr=True optimizer=SGD momentum=0.937 weight_decay=0.0005 \
  &> logs/satellite_e6_b1_hd_a.txt &

# satellite EXP 7-hd-a @ ada-6000
  將影像提高到 1600 訓練並擴增 (包含 109, 109-2, 110, 111)
  遇到 device 序號問題
  ```
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES=1                       # 只暴露實體 GPU#1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ```

## dataset preparation
    # 1. consolidate and convert (already done)
    # python /home/ubuntu/Desktop/Rontgen/dataset_utils/labelme2yolo12seg.py \
    # --input /home/ubuntu/Desktop/Rontgen/dataset/satellite/ \
    # --output /home/ubuntu/Desktop/Rontgen/dataset/satellite/yolodataset \
    # --names /home/ubuntu/Desktop/Rontgen/yolov12/data/satellite-label-names.txt \
    # --val-ratio 0.2 --test-ratio 0.0 --copy

    # 2. offline augmentation (already done)
    # conda run -n yolov12 python /home/ubuntu/Desktop/Rontgen/dataset_utils/seg_augmentation.py \
    # /home/ubuntu/Desktop/Rontgen/dataset/satellite/yolodataset/images/train \
    # /home/ubuntu/Desktop/Rontgen/dataset/satellite/yolodataset/labels/train \
    # /home/ubuntu/Desktop/Rontgen/dataset/satellite/yolodataset/augmented \
    # /home/ubuntu/Desktop/Rontgen/dataset_utils/hyp.yaml \
    # --new_image 2

    # 3. organize augmented folder for YOLO (already done)
    # mkdir -p /home/ubuntu/Desktop/Rontgen/dataset/satellite/yolodataset/augmented/images
    # mkdir -p /home/ubuntu/Desktop/Rontgen/dataset/satellite/yolodataset/augmented/labels
    # mv /home/ubuntu/Desktop/Rontgen/dataset/satellite/yolodataset/augmented/*.png /home/ubuntu/Desktop/Rontgen/dataset/satellite/yolodataset/augmented/images/
    # mv /home/ubuntu/Desktop/Rontgen/dataset/satellite/yolodataset/augmented/*.txt /home/ubuntu/Desktop/Rontgen/dataset/satellite/yolodataset/augmented/labels/

## train @ conda env: yolov12
nohup python /home/ubuntu/Desktop/Rontgen/yolov12/train.py \
  --model /home/ubuntu/Desktop/Rontgen/yolov12/yolov12x-seg.pt \
  --pretrained "" \
  --data /home/ubuntu/Desktop/Rontgen/dataset/satellite/yolodataset/dataset.yaml \
  --project satellite --name e7_b1_hd_a \
  --imgsz 1600 --batch 1 \
  -- \
  device=1 warmup_epochs=8 \
  patience=300 close_mosaic=60  \
  hsv_h=0.000 hsv_s=0.7 hsv_v=0.4 \
  degrees=10 translate=0.1 shear=0.0 perspective=0.0 \
  fliplr=0.0 flipud=0.5 \
  copy_paste=0.0 \
  cos_lr=True optimizer=SGD momentum=0.937 weight_decay=0.0005 \
  &> logs/satellite_e7_b1_hd_a.txt &