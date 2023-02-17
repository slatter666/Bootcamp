## German-English Translation (LSTM with Attention)

#### 1. 数据集准备
- 数据集的介绍在[03_simple_translation](https://github.com/slatter666/Bootcamp/tree/bootcamp/03_simple_translation)中已经详述过
- 运行脚本下载、解压以及处理数据```sh generate.sh```

做完以上工作后dataset目录结构如下
```
dataset
├── origin_data
│   ├── de.dict
│   ├── dev.txt
│   ├── en.dict
│   ├── test.txt
│   └── train.txt
└── raw_data
    ├── test.de
    ├── test.en
    ├── train.de
    ├── train.en
    ├── valid.de
    └── valid.en
```

#### 2. fairseq数据预处理
- 运行脚本对预料进行分词、构建词汇表```sh preprocess.sh```，注意这里其实并没有使用bpe，之后会在transformer中用到

#### 3. 模型搭建及训练
- 此处实现的LSTM with attention模型，encoder使用BiLSTM，decoder使用LSTM，并使用Attention，类似于[03_simple_translation](https://github.com/slatter666/Bootcamp/tree/bootcamp/03_simple_translation)中的advance模型，不过一些细节有所不同。Attention细节参考[Effective approaches to attention-based neural machine translation](https://arxiv.org/pdf/1508.04025.pdf)。
- fairseq对于其他过程已经封装的比较好了，所以这里写完模型模块就可以开始训练了，运行脚本```sh train.sh```开始训练
- 这里训练配置是两块3090，平均一轮1min多一点

#### 4.模型测试
- 解码使用了incremental decoding所以inference非常快，双卡几十秒就能跑完beam search=5的inference
- 运行脚本```sh evaluate.sh```进行测试

#### 5.模型效果
此处仅列举部分参数设置，更多设置请查看train.sh

|        Architecture         | Batch-size  | Epoch |                                  Parameter                                   | BLEU  |
|:---------------------------:|:-----------:|:-----:|:----------------------------------------------------------------------------:|:-----:|
| LSTM seq2seq with Attention | 128 per GPU |  50   |               embed_size=hidden_size=300, layer=2, dropout=0.2               | 27.82 |

