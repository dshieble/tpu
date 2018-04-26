export PYTHONPATH=$PYTHONPATH:$(pwd)
CUDA_VISIBLE_DEVICES=0; python ~/tpu/models/official/resnet/resnet_main.py \
  --use_tpu=false \
  --data_dir=/media/data_cifs/fake_imagenet \
  --model_dir=/media/data_cifs/resnet-tpu-paper-v2_50 \
  --resnet_depth=paper-v2_50 \
  --train_batch_size 256 \
  --eval_batch_size 256 | tee -a resnet-tpu-paper-v2_50

