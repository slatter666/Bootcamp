export CUDA_VISIBLE_DEVICES=4,5
DATA=dataset/processed_data

fairseq-train $DATA \
 --user-dir my_fairseq_module \
 --arch ende_translate --bpe sentencepiece \
 --encoder-embed-dim 300 --encoder-hidden-dim 300 --decoder-embed-dim 300 --decoder-hidden-dim 300 \
 --num-layer 2 --dropout 0.2 \
 --optimizer adam --lr 5e-4 --lr-shrink 0.5 \
 --save-interval 20 --max-sentences 128 --max-epoch 50 \
 --log-file checkpoints/logfile --tensorboard-logdir checkpoints/log --save-dir checkpoints
# BLEU4 = 27.82, 64.3/36.7/22.7/14.5 (BP=0.937, ratio=0.939, syslen=123181, reflen=131141)