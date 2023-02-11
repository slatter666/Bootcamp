## German-English Translation

#### 1. 数据集准备
- 首先下载[数据集](http://cs.stanford.edu/~bdlijiwei/process_data.tar.gz)到data目录下
- 运行解压命令```tar -zxvf process_data.tar.gz```
- 删除压缩包```rm process_data.tar.gz```

做完以上工作后data目录结构如下
```
data
└── process_data
    ├── de.dict
    ├── dev.txt
    ├── en.dict
    ├── test.txt
    └── train.txt
```

简单介绍一下数据集
- 英文词汇表在en.dict，德语词汇表在de.dict
- 数据集分为train.txt、dev.txt、test.txt, 已经预处理过将数据转换为了token形式，其中用```|```隔开，之前的内容是德语，之后的内容是英语
- 由于数据已经处理好了，所以不需要进行数据预处理
- 数据的token是从1开始（而不是从0开始）的，也就是de.dict文件中每行的词对应的token就是它的行号
- 依次运行下列命令可以查看5条翻译句子对，如果想要查看更多可以自行修改for循环
```shell
cd simple
python utils.py
```
#### 2. 模型搭建及训练
这里主要做三个模型进行一下对比
+ simple：基础的seq2seq模型，encoder使用BiLSTM，decoder使用LSTM
+ advanced：进阶的seq2seq模型，encoder使用BiLSTM，decoder使用LSTM，并使用Attention
+ transformer：高阶的seq2seq模型，encoder和decoder均使用transformer

**注意：** 要求达到27的BLEU值会在下一个project中达到，这里所有模型及训练全是手写，不太好实现分布式训练，下个project中会使用fairseq完成所有要求

模型及训练代码放置在对应文件夹中，大致结构如下
```
.
├── model.py
├── run.py
├── run.sh
└── utils.py
```
这里utils.py内容基本一致，run.py内容差别不大，主要的差别model.py中（毕竟不同的模型肯定是不一样的）。相比于simple，advance还计算了ppl、使用beam search进行generate，具体细节可以自行查看

直接运行`sh run.sh`即可，可以自行修改其中的参数，需要注意几点
- 该任务的运行时间较长，请确保自己的硬件够用（这里使用3090单卡，按原参数跑simple一轮大概6min，advance一轮大概8min, transformer一轮大概4min）
- 训练轮数可以适当增大，然后自行判断收敛之后进行early stopping
- 修改run.sh的mode参数选择train进行训练及测试，选择test进行测试

#### 3.模型效果
由于运行时间较久，这里全部设置一组相同的参数进行实验，结果对比如下（可以自行更改参数看看其他参数的效果）

|             |            Architecture            | Batch-size | Epoch |                                  Parameter                                   | BLEU  |
|:-----------:|:----------------------------------:|:----------:|:-----:|:----------------------------------------------------------------------------:|:-----:|
|   simple    |            LSTM seq2seq            |     64     |  15   |               embed_size=hidden_size=300, layer=2, dropout=0.2               | 8.59  |
|   advance   |    LSTM seq2seq with attention     |     64     |  15   |               embed_size=hidden_size=300, layer=2, dropout=0.2               | 24.95 |
| transformer | Transformer seq2seq with attention |     64     |  20   |  embed_size=512,ffn_hid_size=2048,enc_layer=dec_layer=6,n_head=8,dropout=0   | 19.47 |