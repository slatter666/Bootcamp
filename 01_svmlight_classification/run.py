"""
  * FileName: run.py
  * Author:   Slatter
  * Date:     2022/8/17 16:50
  * Description:
  * History:
"""

from utils import svm_light_train, svm_light_test, compute_acc
import argparse

parser = argparse.ArgumentParser(prog="Classification parser")
parser.add_argument("--base", type=str, required=True, help="base directory of dataset")
parser.add_argument("--train", type=str, required=True, help="train dataset filename")
parser.add_argument("--test", type=str, required=True, help="test dataset filename")
parser.add_argument("--mode", type=str, choices=['b', 'm'], default='b', help="classification mode: b for binary classification, m for multi-class classification")
parser.add_argument("--feature", type=str, required=True, help="feature selection pattern: u for unigram, b for bigram, ub for both, g for glove average")
parser.add_argument("--c", type=float, default=1.0, help="regularization parameter")
parser.add_argument("--kernel", type=int, choices=[0,1,2,3], default=0, help="choose kernel: 0: linear, 1: polynomial, 2:radial basis function 3: sigmoid")

group = parser.add_mutually_exclusive_group()
group.add_argument("--news", action="store_true", help="run classification on 20 news group")
group.add_argument("--imdb", action="store_true", help="run classification on imdb movie review")
args = parser.parse_args()

if __name__ == '__main__':
    svm_path = "svm_light"
    base_path = args.base
    if args.feature == "u":
        base_path = args.base + "/unigram"
    elif args.feature == "b":
        base_path = args.base + "/bigram"
    elif args.feature == "ub":
        base_path = args.base + "/ub"
    elif args.feature == "g":
        base_path = args.base + "/glove"

    train_path = args.train
    test_path = args.test
    model_path = "model.txt"
    predict_path = "predict.txt"

    if args.news:
        # 对 20 news group数据集进行分类
        print(f"-----------------------20 News Group classification for feature:{args.feature}-----------------------")
        svm_light_train(svm_path, base_path=base_path, train_file=train_path, model_file=model_path, mode=args.mode, c=args.c, kernel=args.kernel)
        svm_light_test(svm_path, base_path=base_path, test_file=test_path, model_file=model_path, mode=args.mode, predict_path=predict_path)
        print("Accuracy: {:.2f}%".format(compute_acc(base_path=base_path, test_file=test_path, predict_file=predict_path) * 100))
    elif args.imdb:
        print(f"---------------------IMDB Movie Review classification for feature:{args.feature}---------------------")
        # 对 imdb movie review数据集进行分类
        svm_light_train(svm_path, base_path=base_path, train_file=train_path, model_file=model_path, mode=args.mode, c=args.c, kernel=args.kernel)
        svm_light_test(svm_path, base_path=base_path, test_file=test_path, model_file=model_path, predict_path=predict_path, mode=args.mode)
    else:
        print("Warning!!!: Nothing to do, Please check your order --news or --imdb")
