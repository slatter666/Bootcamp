DATA=dataset/raw_data
DEST=dataset/processed_data
fairseq-preprocess \
  --trainpref $DATA/train --validpref $DATA/valid --testpref $DATA/test \
  --source-lang de --target-lang en --bpe subword_nmt \
  --destdir $DEST