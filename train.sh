#!/bin/bash
echo " Running Training EXP"

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --dataset eth --tag sgcn_eth --use_lrschd --num_epochs 300 && echo "eth Launched." &
P0=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --dataset hotel --tag sgcn_hotel --use_lrschd --num_epochs 300 && echo "eth Launched." &
P1=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --dataset univ --tag sgcn_univ --use_lrschd --num_epochs 300 && echo "eth Launched." &
P2=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --dataset zara1 --tag sgcn_zara1 --use_lrschd --num_epochs 300 && echo "eth Launched." &
P3=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --dataset zara2 --tag sgcn_zara2 --use_lrschd --num_epochs 300 && echo "eth Launched." &
P4=$!

wait $P0 $P1 $P2 $P3