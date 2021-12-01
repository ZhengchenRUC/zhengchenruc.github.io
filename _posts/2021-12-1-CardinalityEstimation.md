---
layout: post
title:  "Cardinality Estimation"
date:   2021-12-1
tags: Cardinality
subtitle: "Cardinality Estimation"
description: 'Cardinality Estimation'
color: 'rgb(154,133,255)'
cover: '../images/ce.png'


---



**Cardinality Estimation(基数估计)**



**问题定义**

给定一张关系表 $T$或$R$, 其中包含n个属性(对应表的n个列)$\{A_1, A_2, ..., A_n\}$,  $A_i$的取值范围为$[LB_i, UB_i]$

SQL查询$Q$为：

$$Q=(A_1 \in [LB_1, UB_1] \bigwedge A_2 \in [LB_2, UB_2] \bigwedge ... \bigwedge A_n \in [LB_n, UB_n])$$

返回T中满足查询语句的元组的个数

SQL语句可以表示为：

```*
SELECT COUNT(*)
From R
WHERE Cond_A1 and Cond_A2 ... and Cond_An 
```



**解决方案**



- Traditional Methods:
  - Histogram
  - Sample
  - BayesNet
  - Wander Join/XDB
    - Paper:https://www.cse.ust.hk/~yike/sigmod16.pdf
    - Code:https://github.com/InitialDLab/XDB
  - Pessimistic Cardinality Estimator:
  	- Paper:https://waltercai.github.io/assets/pessimistic-query-optimization.pdf
  	- Code:https://github.com/waltercai/pqo-opensource
- Query-Driven Methods:
  - MSCN
    - Paper:https://arxiv.org/pdf/1809.00677.pdf
    - Code:https://github.com/andreaskipf/learnedcardinalities
  - LW-XGB & LW-NN
    - Paper:http://www.vldb.org/pvldb/vol12/p1044-dutt.pdf
- Data-Driven Methods
  - Bayescard:
    - Paper:https://arxiv.org/pdf/2012.14743.pdf
    - Code:https://github.com/wuziniu/BayesCard
  - Neurocard
    - Paper:https://arxiv.org/pdf/2006.08109.pdf
    - Code:https://github.com/neurocard/neurocard
  - UAE
    - Paper:https://dl.acm.org/doi/10.1145/3448016.3452830
  - DeepDB
    - Paper:https://arxiv.org/pdf/1909.00607.pdf
    - Code:https://github.com/DataManagementLab/deepdb-public
  - FLAT:
    - Paper:https://vldb.org/pvldb/vol14/p1489-zhu.pdf
    - Code[FSPN]:https://github.com/wuziniu/FSPN

