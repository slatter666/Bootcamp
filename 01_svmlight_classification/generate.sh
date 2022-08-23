cp glove840B/glove.840B.300d.txt glove840B/glove.840B.300d.word2vec.txt
sed -i '1 i 2196016 300' glove.840B.300d.word2vec.txt

python generate_data.py --name news \
                        --data_path data/raw_data \
                        --feature u \
                        --save_path data/processed_data

python generate_data.py --name news \
                        --data_path data/raw_data \
                        --feature b \
                        --save_path data/processed_data

python generate_data.py --name news \
                        --data_path data/raw_data \
                        --feature ub \
                        --save_path data/processed_data

python generate_data.py --name news \
                        --data_path data/raw_data \
                        --feature g \
                        --save_path data/processed_data

python generate_data.py --name movie \
                        --data_path data/raw_data \
                        --feature u \
                        --save_path data/processed_data

python generate_data.py --name movie \
                        --data_path data/raw_data \
                        --feature b \
                        --save_path data/processed_data

python generate_data.py --name movie \
                        --data_path data/raw_data \
                        --feature ub \
                        --save_path data/processed_data

python generate_data.py --name movie \
                        --data_path data/raw_data \
                        --feature g \
                        --save_path data/processed_data