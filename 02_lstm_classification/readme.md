## Stanford Sentiment Treebank (SST) 分类

#### 1. 数据集准备
- 首先下载[SST数据集](http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip)，也可以去[官网](https://nlp.stanford.edu/sentiment/code.html)点击右边Dataset Downloads进行下载，这里下载的是第一个文件
- 将压缩包解压之后放到data目录下
- 将上次SVM-Light用到的glove.840B.300d.word2vec.txt文件爱你拷贝到glove840B目录下

做完以上工作之后目录结构如下
```
.
├── data
│   └── stanfordSentimentTreebank
│       ├── datasetSentences.txt
│       ├── datasetSplit.txt
│       ├── dictionary.txt
│       ├── original_rt_snippets.txt
│       ├── README.txt
│       ├── sentiment_labels.txt
│       ├── SOStr.txt
│       └── STree.txt
└── glove840B
          └── glove.840B.300d.word2vec.txt
```

简介一下数据集(可以自行查看README.txt了解数据集)，因为一开始对STree.txt有点懵理解了挺久
1. original_rt_snippets.txt包含了10605段原始句子，其中每段可能包含多个句子，实际用不到
2. dictionary.txt包含了所有短语以及对应的id，中间用|隔开，也就是将短语映射到id
3. sentiment_labels.txt包含短语的id和对应的sentiment label，中间用|隔开，也就是将短语id映射到label，那么根据dictionary.txt和sentiment_labels.txt就可以得到短语到sentiment label的映射。label的值在[0,1]，分为五类[0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]分别对应very negative、negative、neutral、positive、very positive，所以可以自行根据label值将其转换为0、1、2、3、4类别标签
4. SOStr.txt和STree.txt是对语法树的表示。其中SOStr.txt可以视为一个tokenizer确定好每句话对应的tokens，STree.txt就是将整句话变成语法树的形式，语法树采用父指针表示法来构建，示例结构如图。
5. datasetSentences.txt包含了句子index以及句子，中间用tab隔开，其中每个句子token之间用空格隔开，是整个数据集包含train、dev、test（同时该文件和SOStr.txt中每个句子也是一一对应关系）
6. datasetSplit.txt包含了句子index以及对应的label，label表示所属数据集：1属于train、2属于test、3属于dev。当然你可以自行划分数据集，但是如果你要用SST做实验发论文做比较那么就需要根据它的方式来划分，算是公平比较吧

#### 2. 数据预处理
由于datasetSentences.txt中的句子可能会存在“AmÃ©lie”这种字符串，里面存在一些法语符号以及一些标签“-LRB- A -RRB-”和dictionary对不上，所以用SOStr.txt恢复出来的句子做映射。因此需要用到的文件有：dictionary.txt、sentiment_labels.txt、SOStr.txt、STree.txt、datasetSplit.txt

可以直接运行下面脚本得到最终处理好的数据，处理过程分为四步：分割数据集、根据trainSentence.txt生成训练集词汇表、根据语法树生成短语训练集train.txt、最后生成数据集label
```shell
sh generate.sh
```
可以运行以下命令查看根据语法树构建句子所有phrase的示例
```shell
python utils.py
```

用SOStr.txt恢复出来的句子总数和论文是能完全匹配的，train/dev/test二分类分别占6920/872/1821， 五分类分别占8544/1101/2210。数据预处理完成后data目录结构如下
```
data
├── binary
│   ├── devlabel.txt
│   ├── devTree.txt
│   ├── dev.txt
│   ├── testlabel.txt
│   ├── testTree.txt
│   ├── test.txt
│   ├── trainlabel.txt
│   ├── trainSentence.txt
│   ├── trainTree.txt
│   ├── train.txt
│   └── vocab.txt
├── fine_grained
│   ├── devlabel.txt
│   ├── devTree.txt
│   ├── dev.txt
│   ├── testlabel.txt
│   ├── testTree.txt
│   ├── test.txt
│   ├── trainlabel.txt
│   ├── trainSentence.txt
│   ├── trainTree.txt
│   ├── train.txt
│   └── vocab.txt
└── stanfordSentimentTreebank
    ├── datasetSentences.txt
    ├── datasetSplit.txt
    ├── dictionary.txt
    ├── original_rt_snippets.txt
    ├── README.txt
    ├── sentiment_labels.txt
    ├── SOStr.txt
    └── STree.txt

```
在后续过程中会用到的文件主要是train.txt、trainlabel.txt、dev.txt、devlabel.txt、test.txt、testlabel.txt、vocab.txt，其他文件主要是中间文件。训练使用的是训练集句子的所有phrase，所以需要根据语法树对训练集数据做一个phrase的恢复，而验证集、测试集是不需要构建所有phrase的（做过实验，如果dev和test也分为phrase那么不使用glove五分类任务上的正确率可以达到67%）。

#### 3. 训练模型

模型使用2-layer Bidirectional LSTM，根据论文使用最后的LSTM hidden state进行分类，具体模型结构查看```model.py```代码

运行以下脚本可以开始训练(训练代码查看```run.py```)
```shell
sh run.sh
```
这里显存不够所以使用cpu训练（如果你想使用gpu训练请确保你的显存大于8G，大概需要6、7个G），参数均设置得和原论文一样，可以自行在run.sh中调整参数。

#### 4. 分类效果

|              |                   Architecture                    | Batch-size | Epoch | Accuracy |
|:------------:|:-------------------------------------------------:|:----------:|:-----:|:--------:|
| Fine-grained |            2-layer Bidirectional LSTM             |     25     |  15   |  50.63   |
|    Binary    |            2-layer Bidirectional LSTM             |     25     |  10   |  86.44   |
