export CUDA_VISIBLE_DEVICES=4
python run.py \
  --batch 64 --epochs 30 --lr 1e-4 --dropout 0\
  --embed-size 512 --ffn-hid-size 2048 --encoder-layer 6 --decoder-layer 6\
  --max-len 50 --mode test  # BLEU: 19.47
