export CUDA_VISIBLE_DEVICES=4,5
DATA=dataset/processed_data
fairseq-generate $DATA \
  --user-dir my_fairseq_module \
  --path checkpoints/checkpoint_best.pt\
  --max-sentences 128 --beam 5 --remove-bpe