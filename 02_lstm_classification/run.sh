

python run.py --mode f \
              --batch 25 \
              --epoch 10 \
              --layers 2 \
              --dropout 0.5 \
              --pretrain glove840B/glove.840B.300d.word2vec.txt \
              --device cpu