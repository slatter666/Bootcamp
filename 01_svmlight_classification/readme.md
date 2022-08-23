## SVM-Light分类

#### 1. 预先下载好所需的工具

- 将glove.840B.300d.txt预训练词向量文件放在glove840B文件夹下，词向量地址：[Glove840B](https://nlp.stanford.edu/data/glove.840B.300d.zip)
- 将svm_light的svm_learn.exe、svm_classify.ext、svm_multiclass_learn.exe、svm_multiclass_classify.exe放在svm_light文件夹下，程序下载地址：[svm_light](https://www.cs.cornell.edu/people/tj/svm_light/)、[svm_multiclass](https://www.cs.cornell.edu/people/tj/svm_light/svm_multiclass.html)
- 预先创建data/processed_data、data/raw_data用于之后存放数据

做完以上工作之后目录结构大致如下

```
.
|-- data
|   |-- processed_data
|   `-- raw_data
|-- glove840B
    `- glove.840B.300d.txt
|-- svm_light
    |-- svm_classify.exe
    |-- svm_learn.exe
    |-- svm_multiclass_classify.exe
    `-- svm_multiclass_learn.exe
```

**注意：**本次运行环境是在Windows，所以可执行文件为.exe格式，在Linux环境下载对应的文件即可，不需要修改代码（应该不用修改，没具体试过，如果分类报错可以检查一下utils.py）。另一个可预想的Linux报错应该是文件路径，因为Windows和Linux对路径的表示斜杠使用方式不一样。



#### 2. 进行数据预处理

运行脚本对20 News Group数据集和IMDB Movie Review数据集进行数据处理以适应svm_light程序所需的数据输入格式（Glove840B大概5个G，加载会很久所以程序运行时间会较长）

```shell
sh generate.sh
```

完成后数据目录结构如下（data目录结构）

```
data
|-- processed_data
|   |-- movie
|   |   |-- bigram
|   |   |   |-- test.txt
|   |   |   |-- train.txt
|   |   |   `-- words.txt
|   |   |-- glove
|   |   |   |-- test.txt
|   |   |   |-- train.txt
|   |   |   `-- words.txt
|   |   |-- ub
|   |   |   |-- test.txt
|   |   |   |-- train.txt
|   |   |   `-- words.txt
|   |   `-- unigram
|   |       |-- test.txt
|   |       |-- train.txt
|   |       `-- words.txt
|   `-- news
|       |-- bigram
|       |   |-- test.txt
|       |   |-- train.txt
|       |   `-- words.txt
|       |-- glove
|       |   |-- test.txt
|       |   |-- train.txt
|       |   `-- words.txt
|       |-- ub
|       |   |-- test.txt
|       |   |-- train.txt
|       |   `-- words.txt
|       `-- unigram
|           |-- test.txt
|           |-- train.txt
|           `-- words.txt
`-- raw_data
    `-- 20news-bydate_py3.pkz
```



#### 3. 进行分类

运行脚本放在run.sh中，可以自行注释脚本逐个运行，这样对于每个分类效果看起来会更清晰一点

```shell
sh run.sh
```

完成后数据目录结构如下

```
data
|-- processed_data
|   |-- movie
|   |   |-- bigram
|   |   |   |-- model.txt
|   |   |   |-- predict.txt
|   |   |   |-- test.txt
|   |   |   |-- train.txt
|   |   |   `-- words.txt
|   |   |-- glove
|   |   |   |-- model.txt
|   |   |   |-- predict.txt
|   |   |   |-- test.txt
|   |   |   |-- train.txt
|   |   |   `-- words.txt
|   |   |-- ub
|   |   |   |-- model.txt
|   |   |   |-- predict.txt
|   |   |   |-- test.txt
|   |   |   |-- train.txt
|   |   |   `-- words.txt
|   |   `-- unigram
|   |       |-- model.txt
|   |       |-- predict.txt
|   |       |-- test.txt
|   |       |-- train.txt
|   |       `-- words.txt
|   `-- news
|       |-- bigram
|       |   |-- model.txt
|       |   |-- predict.txt
|       |   |-- test.txt
|       |   |-- train.txt
|       |   `-- words.txt
|       |-- glove
|       |   |-- model.txt
|       |   |-- predict.txt
|       |   |-- test.txt
|       |   |-- train.txt
|       |   `-- words.txt
|       |-- ub
|       |   |-- model.txt
|       |   |-- predict.txt
|       |   |-- test.txt
|       |   |-- train.txt
|       |   `-- words.txt
|       `-- unigram
|           |-- model.txt
|           |-- predict.txt
|           |-- test.txt
|           |-- train.txt
|           `-- words.txt
`-- raw_data
    `-- 20news-bydate_py3.pkz
```



#### 4. 20 News Group数据集分类效果

基本是将c值调的越大越好，当然训练时间也会变得较久（此处调的参数并不保证最优，不过再继续调下去波动也很微小，如果要求尽快出结果可以调小一个数量级）

|                | c(regularization) | kernel | accuracy |
| :------------: | :---------------: | :----: | :------: |
|    unigram     |      1000000      | linear |  71.00%  |
|     bigram     |       5000        | linear |  57.40%  |
| unigram+bigram |       5000        | linear |  70.59%  |
|     glove      |       50000       | linear |  66.34%  |



#### 5. IMDB Movie Review数据集分类效果

采用linear kernel收敛速度较快而且效果还挺不错所以全部采用linear kernel（对glove采用了rbf试了一下但实际上F1也就高了0.5%，大致是差不太多的）

|                | c(regularization) | kernel | accuracy | precision | recall |   F1   |
| :------------: | :---------------: | :----: | :------: | :-------: | :----: | :----: |
|    unigram     |        0.3        | linear |  87.83%  |  87.60%   | 88.14% | 87.87% |
|     bigram     |        1.0        | linear |  84.15%  |  84.64%   | 83.45% | 84.04% |
| unigram+bigram |        1.0        | linear |  88.62%  |  88.82%   | 88.37% | 88.59% |
|     glove      |                   |  rbf   |  85.83%  |  85.67%   | 86.06% | 85.86% |



#### 6.运行可能会存在的问题

1. 在Linux下运行会存在路径表示问题，需要自行修改
2. Windows下使用open操作似乎并不能自行递归创建文件，需要自行把所有文件目录创建好（所以Windows中的小bug很多，之后实验环境设定将全为Linux）

**致谢：** 在进行多分类时查到了[这篇文章](https://gohom.win/2015/08/12/svmlight/)才发现SVM-Multiclass

