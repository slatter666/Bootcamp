export CUDA_VISIBLE_DEVICES=4
DATA=dataset/processed_data
fairseq-train $DATA \
  --user-dir my_fairseq_module \
  --task nmt_task --arch nmt \
  --optimizer adam\
  --lr 2.5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --max-tokens 1000 \
  --max-epoch 20 --save-interval 10 \
  --save-dir checkpoints --log-file checkpoints/logfile --tensorboard-logdir checkpoints/logs \
  --skip-invalid-size-inputs-valid-test \
  --fp16
# Generate test with beam=4: BLEU4 = 32.50, 63.0/38.7/26.2/18.2 (BP=0.990, ratio=0.990, syslen=141130, reflen=142528)