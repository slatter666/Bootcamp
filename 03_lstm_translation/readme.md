## German-English Translation

#### 1. 数据集准备
- 首先下载[数据集](http://cs.stanford.edu/~bdlijiwei/process_data.tar.gz)到data目录下
- 运行解压命令```tar -zxvf process_data.tar.gz```
- 删除压缩包```rm process_data.tar.gz```

做完以上工作后data目录结构如下
```
.
└── process_data
    ├── de.dict
    ├── dev.txt
    ├── en.dict
    ├── test.txt
    └── train.txt
```

简单介绍一下数据集
- 英文词汇表在en.dict，德语词汇表在de.dict
- 数据集分为train.txt、dev.txt、test.txt, 已经预处理过将数据转换为了token形式，其中用|隔开，之前的内容是德语，之后的内容是英语
- 由于数据已经处理好了，所以不需要进行数据预处理
- 数据的token是从1开始（而不是从0开始）的，也就是de.dict文件中每行的词对应的token就是它的行号
- 依次运行下列命令可以查看5条翻译句子对，如果想要查看更多可以自行修改main中的nums
```shell
cd simple
python utils.py
```
#### 2. 模型搭建及训练
这里主要做两个模型进行一下对比
+ simple：基础的seq2seq模型，encoder使用BiLSTM，decoder使用LSTM
+ advanced：进阶的seq2seq模型，encoder使用BiLSTM，decoder使用LSTM，并使用Attention

模型及训练代码放置在对应文件夹中，大致结构如下
```
.
├── best.pth
├── model.py
├── run.py
├── run.sh
└── utils.py
```
这里utils.py其实内容差距不大，有些细节上的差别但是由于simple本来就是一个练手版本所以不太想改了，建议查看advanced/utils.py

其中直接运行`sh run.sh`即可，可以自行修改其中的参数，需要注意几点
- simple模型可以看看或者跑一两个epoch，没必要去跑完因为只是用来了解一下最原始的seq2seq怎么做的，而且还很费时
- 跑完模型之后记得注释掉train()然后就可以直接测试了
- 在训练的时候句子长度并不算很长，所以max-len可以调小一点，但是test会有很长的句子需要max-len调到超过180,这样会在训练阶段节省一些时间

#### 3.模型效果
重点不在simple seq2seq，这里仅记录加上了attention的模型效果

