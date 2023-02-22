cd dataset
wget http://cs.stanford.edu/~bdlijiwei/process_data.tar.gz
tar -xzvf process_data.tar.gz
rm process_data.tar.gz
mv process_data origin_data
mkdir raw_data

cd ../preprocess_data
python generate.py
python tokenize_data.py