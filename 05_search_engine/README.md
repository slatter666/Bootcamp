## 搜索引擎项目

### 背景介绍

#### 简介

搜索引擎主要由三部分组成： crawler 爬取网页， indexer 建立索引， searcher 完成检索。此次任务不考虑 crawler 部分，根据本地存取的文档建立索引并可接收 query 完成检索。

#### 搜索引擎工作流程

流程主要分为两部分： 离线部分和在线部分。

1. 离线部分， 建立正排表和倒排表。
2. 在线部分，通过 query 进行查询。

### 数据集介绍

数据分为两部分：
1. query_list.txt，是3140个不同的query
2. news_title,txt，是与这些query相关的新闻标题

路径： 在 GPU 服务器上，位置是 `shannon-bootcamp/05_search_engine`

### 要求

实现搜索引擎，输入是一个 query，输出是与其最相关的十个新闻标题。可与 `https://ask.shannonai.com:4443/#/` 结果比对

### 参考文献
http://web.stanford.edu/class/cs276/

### 使用介绍
输入任意字符（输入0结束搜索）进行随机query的搜索，根据词频来进行rank，感觉效果还不错。查找速度较快，能够满足实时要求。