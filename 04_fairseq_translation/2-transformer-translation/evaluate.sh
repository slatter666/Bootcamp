export CUDA_VISIBLE_DEVICES=4
DATA=dataset/processed_data
fairseq-generate $DATA \
  --user-dir my_fairseq_module \
  --task nmt_task \
  --path checkpoints/checkpoint_best.pt \
  --max-sentences 128 --beam 4 --remove-bpe \
  --skip-invalid-size-inputs-valid-test
