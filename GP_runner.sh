#!/bin/bash
#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH -o out.log
#SBATCH -e err.logpip freeze > requirements.txt
module load nvidia/cuda/10.0
module load pytorch/1.0_python3.7_gpu
python src/train_CME.py \
  --do_train \
  --do_predict \
  --bert_model_path ../pre_ckpts/my_mengzi_9 \
  --data_dir datasets/split_data \
  --batch_size 16 \
  --epoch 32 \
  --encoder_learning_rate 2e-5 \
  --decoder_learning_rate 4e-4 \
  --do_rdrop \
  --rdrop_alpha 1 