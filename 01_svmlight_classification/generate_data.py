"""
  * FileName: generate_data
  * Author:   Slatter
  * Date:     2022/8/21 14:14
  * Description: 
  * History:
"""
import argparse
import gensim
from torchtext.datasets import IMDB
from DataProcesser import DataProcesser
from sklearn.datasets import fetch_20newsgroups
from torchtext.data.functional import to_map_style_dataset

parser = argparse.ArgumentParser(prog="processing data")
parser.add_argument("--name", type=str, required=True, choices=["news", "movie"],
                    help="name of dataset we need to process, choose news or movie")
parser.add_argument("--data_path", type=str, default="data/raw_data", help="data source, default: data/raw_data")
parser.add_argument("--feature", type=str, required=True, choices=["u", "b", "ub", "g"],
                    help="decide feature selection:u for unigram, b for bigram, ub for both, g for glove average")
parser.add_argument("--save_path", type=str, default="data/processed_data",
                    help="decide which place to store processed data, default: data/processed_data")
args = parser.parse_args()

if __name__ == '__main__':
    # 有可能生成的数据会少一些，因为有的句子会存在整个句子都是停用词（或者是个空字符串的情况）导致该句直接被移除
    data_path = args.data_path
    save_path = args.save_path
    feature = args.feature

    if args.name == "news":
        data_path = "data/raw_data"
        save_path = "data/processed_data"
        train_data = fetch_20newsgroups(data_home=data_path, subset="train",
                                        remove=("headers", "footers", "quotes"))  # 11314
        test_data = fetch_20newsgroups(data_home=data_path, subset="test",
                                       remove=("headers", "footers", "quotes"))  # 7532

        processer = DataProcesser(name="20news", train_data=train_data['data'], train_label=train_data['target'],
                                  test_data=test_data['data'], test_label=test_data['target'], feature="ub",
                                  save_path=save_path)
        # 处理20 news group数据集
        train_data = fetch_20newsgroups(data_home=data_path, subset="train", remove=("headers", "footers", "quotes"))
        test_data = fetch_20newsgroups(data_home=data_path, subset="test", remove=("headers", "footers", "quotes"))

        # 多分类类别必须为正整数，所以将train_data['target']和test_data['target']都加1
        processer = DataProcesser(name="news", train_data=train_data['data'], train_label=train_data['target'] + 1,
                                  test_data=test_data['data'], test_label=test_data['target'] + 1, feature=feature,
                                  save_path=save_path)
        processer.process()
    elif args.name == "movie":
        # 处理imdb movie review数据集
        train_dataset = to_map_style_dataset(IMDB(split="train"))
        test_dataset = to_map_style_dataset(IMDB(split="test"))

        train_data, train_label, test_data, test_label = [], [], [], []
        for data in train_dataset:
            sentiment, text = data
            train_data.append(text)
            # 将评价极性转换为数字 neg->-1 pos->+1
            if sentiment == "pos":
                train_label.append("+1")
            else:
                train_label.append("-1")

        for data in test_dataset:
            sentiment, text = data
            test_data.append(text)
            # 将评价极性转换为数字 测试集好像不太一样需要都转换为带符号的形式 neg->-1 pos->+1
            if sentiment == "pos":
                test_label.append("+1")
            else:
                test_label.append("-1")
        processer = DataProcesser(name="movie", train_data=train_data, train_label=train_label, test_data=test_data,
                                  test_label=test_label, feature=feature, save_path=save_path)
        processer.process()
    else:
        raise Exception("You have to type correct name of dataset: news or movie")

    print(f"{args.name} Data Generation for feature:{feature} Finished")  # 输出一点反馈
