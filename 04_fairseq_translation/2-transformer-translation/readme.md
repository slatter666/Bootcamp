## German-English Translation (Transformer)

#### 1. 数据集准备
- 数据集的介绍在[03_simple_translation](https://github.com/slatter666/Bootcamp/tree/bootcamp/03_simple_translation)中已经详述过
- 运行脚本下载、解压以及处理数据```sh download.sh```

做完以上工作后dataset目录结构如下
```
dataset
├── origin_data
│   ├── de.dict
│   ├── dev.txt
│   ├── en.dict
│   ├── test.txt
│   └── train.txt
├── raw_data
│   ├── test.de
│   ├── test.en
│   ├── train.de
│   ├── train.en
│   ├── valid.de
│   └── valid.en
└── tokenized
    ├── test.json
    ├── train.for_bpe.src
    ├── train.for_bpe.tgt
    ├── train.json
    └── valid.json
```

#### 2. fairseq数据预处理
- 在transformer中我们使用fastBPE进行分词，首先从[fastBPE](https://github.com/glample/fastBPE)上下载代码
```shell
git clone https://github.com/glample/fastBPE
```
- 然后按照fastBPE的readme进行编译，之后会生成一个可执行文件```fast```
```shell
g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
```
- 参考readme调用可执行文件学习bpe
```shell
fastBPE/fast learnbpe 40000 dataset/tokenized/train.for_bpe.src dataset/tokenized/train.for_bpe.tgt > codes
```
- 将bpe apply到train上
```shell
fastBPE/fast applybpe train.src.40000 dataset/tokenized/train.for_bpe.src codes
fastBPE/fast applybpe train.tgt.40000 dataset/tokenized/train.for_bpe.tgt codes
```
- 获取词汇表
```shell
fastBPE/fast getvocab train.src.40000 > vocab.src.40000
fastBPE/fast getvocab train.tgt.40000 > vocab.tgt.40000
```
- 得到了codes和vocab之后将其应用到所有数据上，这个地方用python代码进行处理，参考readme安装fastbpe的python包,然后将我们需要的文件放到preprocess文件夹中
```shell
python setup.py install
mv codes vocab.src.40000  vocab.tgt.40000 preprocess
rm train.src.40000  train.tgt.40000

cd preprocess
python apply_bpe.py
python build_dict.py
```

#### 3. 模型搭建及训练
- 此处实现的Transformer seq2seq模型，模型以及任务代码均放置在my_fairseq_module中，代码细节这里就不详述了
- 需要确保自己项目文件下已经创建好了checkpoints文件，没有则执行命令```mkdir checkpoints```进行创建
- 运行脚本```sh train.sh```开始训练
- 这里训练配置是一块3090，多卡要报错之后再解决，只要batch size设置的合理平均一轮也就一两分钟

#### 4.模型测试
- 解码单卡几十秒就能跑完beam search=4的inference，总的来说用bpe会比不用bpe提升好几个点
- 运行脚本```sh evaluate.sh```进行测试

#### 5.模型效果
- 下面记录跑过的一些实验，以及关于调参的一些经验
- 为了达到同一个结果，在增大batch size的时候需要等比例增大learning rate(比如batch size从2000增大到4000，那么学习率也需要从5e-4增大到1e-3)
- transformer的训练必须要使用warmup，否则会无法收敛，warmup按照论文设置为4000比较合适
- batch size在合理范围内设置得稍微大一点，这样训练就会更快，因为不用做那么多次反向传播
- 这里Model Parameter并没有具体调整进行对比，如果有兴趣可以将transformer结构的内部参数调一下看看效果

| Architecture | learning rate | max-tokens | warmup updates | Epoch | Model Parameter | BLEU  |
|:------------:|:-------------:|:----------:|:--------------:|:-----:|:---------------:|:-----:|
| Transformer  |     5e-4      |    6000    |      4000      |  20   |     Default     | 28.35 |
| Transformer  |     5e-4      |    4000    |      4000      |  20   |     Default     | 31.86 |
| Transformer  |     5e-4      |    2000    |      4000      |  20   |     Default     | 32.38 |
| Transformer  |    2.5e-4     |    1000    |      4000      |  20   |     Default     | 32.50 |
| Transformer  |     1e-3      |    4000    |      4000      |  20   |     Default     | 31.85 |
| Transformer  |     5e-4      |    4000    |      2000      |  20   |     Default     | 31.81 |
| Transformer  |     5e-4      |    4000    |      8000      |  20   |     Default     | 31.01 |
| Transformer  |     5e-4      |    4000    |     12000      |  20   |     Default     | 31.11 |




