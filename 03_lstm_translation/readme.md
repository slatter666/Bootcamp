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
- 运行`python utils.py`可以看5条实际翻译数据，修改`data.check_sentence(num)`中的num以查看更多数据

#### 2. 训练模型

