# unigram 20 news group
python run.py --news \
              --base data/processed_data/news \
              --train train.txt \
              --test test.txt \
              --mode m \
              --feature u \
              --c 1000000

# bigram 20 news group
python run.py --news \
              --base data/processed_data/news \
              --train train.txt \
              --test test.txt \
              --mode m \
              --feature b \
              --c 5000

# ub 20 news group
python run.py --news \
              --base data/processed_data/news \
              --train train.txt \
              --test test.txt \
              --mode m \
              --feature ub \
              --c 5000

# glove 20 news group
python run.py --news \
              --base data/processed_data/news \
              --train train.txt \
              --test test.txt \
              --mode m \
              --feature g \
              --c 50000

# unigram imdb movie review
python run.py --imdb \
              --base data/processed_data/movie \
              --train train.txt \
              --test test.txt \
              --feature u \
              --c 0.3

# bigram imdb movie review
python run.py --news \
              --base data/processed_data/movie \
              --train train.txt \
              --test test.txt \
              --feature b

# ub imdb movie review
python run.py --news \
              --base data/processed_data/movie \
              --train train.txt \
              --test test.txt \
              --feature ub

# glove imdb movie review
python run.py --news \
              --base data/processed_data/movie \
              --train train.txt \
              --test test.txt \
              --feature g \
              --kernel 2