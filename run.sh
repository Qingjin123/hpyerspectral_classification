#! /bin/zsh
#
echo 'Begin ~~~~'
conda activate pytorch
EPOCH = 1000
LR = 0.0001
while i >100; do
	echo i
	python main.py --epoch EPOCH --lr LR
done
