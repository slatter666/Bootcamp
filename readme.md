# NLP入门bootcamp

### 代码管理要求：

用gitlab或者github进行代码version管理。每发一次实验之前，需要在 github/wiki/gitlab提交 (commit id, 运行脚本，结果)，结果在实验之后补充

 

### 课程要求

自学斯坦福 CS224N课程，做相关作业。



### 英语练习

上coursera的课，这里可以根据自己现有的英语基础上不同的课，比如稍微基础点的是Duke的[English Composition](https://www.coursera.org/learn/english-composition)，进阶一点的就是Stanford的[Scientific Writing](https://www.coursera.org/learn/sciwrite)



### NLP入门Bootcamp:

1. 学会使用SVM进行分类

   运用[SVM_light](http://svmlight.joachims.org/)开源包分别对20newsgroup, imdb movie dataset 进行分类。对比运用 unigram, bigram, unigram+bigram 以及Glove 840B词向量平均 作为feature的分类结果。

2. 使用双向LSTM对于 Stanford Sentiment Treebank (SST) 进行分类

   参考[Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](https://arxiv.org/pdf/1503.00075.pdf). 复现论文中第6页Table2中BiLSTM结果 2-layer Bidirectional LSTM 48.5 (1.0) 87.2 (1.0)
     

3. 机器翻译分别实现 LSTM attention和transformers

   机器翻译分别实现 LSTM attention（Effective approaches to attention-based neural machine translation） 和 transformers 在german-english 翻译 （[数据下载地址](http://cs.stanford.edu/~bdlijiwei/process_data.tar.gz)，LSTM 需要到 bleu值27以上，transformers 最好可以超过30。

4. 复现BERT在GLUE，NLI以及NER上结果

 

 

 

 