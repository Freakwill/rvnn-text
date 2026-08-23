# rvnn-text — 递归神经网络 (RvNN) 文本生成

一个基于 **PyTorch** 的**递归神经网络（Recursive Neural Network, RvNN）**实现，以**文本生成**为演示，
并阐释 RvNN 与**文法分析（parsing）**之间的内在联系。

- **不是 RNN**：RNN（循环神经网络）把文本当作一条**线性序列**从左到右处理；
  RvNN 则把文本当作一棵**句法树（parse tree）**，自底向上递归地组合子节点向量。
- **文法约束生成**：解码器在**上下文无关文法（CFG）**的约束下自顶向下展开，因此生成的句子**永远合乎语法**。
- 纯 PyTorch 实现，支持 CUDA / MPS（Apple Silicon）/ CPU，设备自动选择。

---

## 目录

- [RvNN 原理](#rvnn-原理)
- [与文法分析的联系](#与文法分析的联系)
- [应用](#应用)
- [安装](#安装)
- [使用方法](#使用方法)
- [演示结果](#演示结果)
- [项目结构](#项目结构)
- [数据来源](#数据来源)
- [扩展方向](#扩展方向)

---

## RvNN 原理

### 递归 vs 循环：结构决定了模型

RNN（Recurrent Neural Network）隐含地假设输入是一个**链式结构**——每个词只依赖它左边的词：

```
RNN:  w1 → w2 → w3 → ... → wn     (线性序列)
```

而自然语言的句法结构是**树状的**，不是线性的。句子 *"the happy cat sees a dog"* 的句法分析树是：

```
            S
          /   \
        NP     VP
       / \     / \
     Det NOM Verb  NP
      |  / \   |   / \
     the Adj NOM sees Det NOM
          |  / \      |   |
        happy Noun    a  Noun
               |          |
              cat        dog
```

RvNN（Recursive Neural Network）沿着这棵树**递归**地组合向量：叶子（单词）先有各自的词向量，然后每
个内部节点把子节点的向量组合成父节点的向量。**同一个组合函数在每一个内部节点反复使用**——这正是
"递归"的含义（recursion，而非 recurrence）：

```
p = tanh( W_left · c_left + W_right · c_right + b )      # 二叉节点
p = tanh( W_unary · c + b_unary )                        # 一元节点
```

其中 `c_left`、`c_right` 是子节点向量，`p` 是父节点向量，`W_*`、`b` 是**全树共享**的参数。

### 为什么是树而不是序列？

1. **组合语义（compositionality）**：短语的含义由部分的含义与组合方式决定——这是形式语义学
   （Frege 的语义组合原则）的核心思想，也是 RvNN 的归纳偏置。
2. **长距离依赖**：对平衡树，从叶子到根的路径长度是 `O(log n)`，而 RNN 中两个相距很远的词之间隔着
   `O(n)` 步，梯度容易消失。树结构天然缩短了信息传播的路径。
3. **结构化输出**：RNN 只能输出序列；RvNN 天然适合"输入是树 / 输出是树"的任务（如句法分析、
   程序抽象语法树）。

### 本项目中的模型

`rvnn_text.model.RvNNText` 是一个**递归自编码器（recursive autoencoder）**：

- **编码器（自底向上）**：叶子 = 词向量；内部节点 = 共享组合函数 `tanh(W_l h_l + W_r h_r + b)`，
  最终整句话被压缩为一个向量 `h_root`。
- **解码器（自顶向下）**：从一个可学习的起始向量出发，在每个内部节点 (a) 预测一条**产生式规则**
  （如 `NP → Det NOM`），(b) 把父向量**投影**为各内部子节点的向量并递归，直到到达词性叶子时采样单词。
- **训练目标**：
  1. **规则交叉熵**：用节点向量预测该节点的产生式规则（监督信号，训练编码器）；
  2. **单词交叉熵**：用**父节点**向量预测其词性子节点的单词（监督信号）；
  3. **重构损失**：解码投影应能近似还原编码器的自底向上向量（**目标 detach**，只训练解码投影，
     避免二者相互拉扯导致向量塌缩）。

关键细节：单词预测以**父节点向量**为条件（而非叶子自身的向量），否则"用词向量预测它自己对应的
词"是平凡退化问题。

---

## 与文法分析的联系

RvNN 与文法分析（parsing）的联系是**结构性的、不可分割的**：

1. **句法树就是 RvNN 的输入结构**。RvNN 处理的"树"，正是**上下文无关文法（CFG）**推导出的
   **成分句法树（constituency parse tree）**：`S → NP VP`、`NP → Det NOM`、`NOM → Adj NOM` 等产生式
   规则定义了树的形状，而 RvNN 沿着这棵树组合向量。**没有文法就没有这棵树，也就没有 RvNN 的递归结构。**

2. **RvNN 可以被当作一个"神经句法分析器"**。本项目的解码器在生成时预测每个节点的产生式规则；
   反向地，给定一棵树的向量表示，`evaluate()` 会度量"从向量还原出每个节点的产生式规则"的准确率。
   实验结果显示规则预测准确率达到 **100%**——也就是说，RvNN 在向量空间中**内化了文法**。

3. **递归 = 组合性**。文法的递归（`NOM → Adj NOM`）对应 RvNN 的递归组合。语法规定了"哪些成分可以
   组合成什么"，RvNN 学会了"如何把子向量组合成父向量"。

4. **经典文献**：Richard Socher 等人的工作正是把递归神经网络建立在此之上——
   - *Parsing Natural Scenes and Natural Language with Recursive Neural Networks* (ICML 2011)
   - *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank* (EMNLP 2013)，
     即著名的 **Stanford Sentiment Treebank（SST）**，其情感标签就标注在成分句法树的每个节点上。

---

## 应用

- **情感分析**：SST 在句法树的每个短语节点标注情感，RvNN 自底向上组合得到整句情感（比"词袋"或
  RNN 更能刻画否定、修饰等组合语义）。
- **关系抽取 / 复述检测**：用句法树的组合向量计算短语或句子的语义相似度。
- **场景解析**：Socher 2011 将图像分割成区域，按区域树递归组合做语义分割。
- **代码 / 程序分析**：程序的抽象语法树（AST）同样是树，RvNN 可用于代码向量化、缺陷检测。
- **文法引导的生成**：在语法约束下生成结构化文本（本项目即为最小示例），常用于程序合成、SQL、
  公式生成等"语法必须正确"的场景。

---

## 安装

```bash
# 克隆并进入项目
git clone https://github.com/<your-name>/rvnn-text.git
cd rvnn-text

# 安装依赖（torch 已装可跳过）
pip install -r requirements.txt

# 以可编辑模式安装（可选，提供 rvnn-train / rvnn-generate / rvnn-demo 命令）
pip install -e .
```

依赖：`torch>=2.0`、`fire`（CLI）、`pytest`（测试）。

---

## 使用方法

### 1. 一键演示（推荐先跑这个）

```bash
python -m rvnn_text.demo
```

会依次展示：文法定义 → 采样出的句法树 → 训练过程 → 文法学习准确率 → 文法约束生成 → 编码-解码
往返（round-trip）。可调参数：

```bash
python -m rvnn_text.demo --num_sentences 1500 --epochs 20 --n_samples 6
```

### 2. 训练

```bash
python -m rvnn_text.train --num_sentences 4000 --epochs 30 --dim 64
```

| 参数 | 含义 | 默认 |
|---|---|---|
| `num_sentences` | 从文法采样的训练句子数 | 4000 |
| `epochs` | 训练轮数 | 30 |
| `dim` | 词向量/隐层维度 | 64 |
| `lr` | 学习率 | 1e-3 |
| `recon_weight` | 重构损失权重 | 1.0 |
| `out_dir` | checkpoint 输出目录 | checkpoints |

训练结束后模型保存在 `checkpoints/rvnn_text.pt`。

### 3. 生成

```bash
# 从先验采样 8 句，并打印每句的句法树
python -m rvnn_text.generate --checkpoint checkpoints/rvnn_text.pt --n 8

# 贪心解码（每步取概率最大的规则/单词）
python -m rvnn_text.generate --greedy --n 5

# 编码一句话 → 从它的向量重新解码（round-trip / 复述演示）
python -m rvnn_text.generate --from_sentence "the happy cat sees a dog"
```

关键参数：`temperature`（采样温度，越大越随机）、`seed_noise`（起始向量的高斯噪声，控制多样性）、
`max_depth`（最大树深度，限制右递归 `NOM → Adj NOM` 造成的无限展开）。

### 4. 编程接口

```python
from rvnn_text import Grammar, RvNNText, make_corpus, get_device, set_seed
from rvnn_text.train import train_model

set_seed(42)
grammar = Grammar()                      # 使用内置英文文法
corpus = make_corpus(grammar, 4000)      # 采样句法树
model, _ = train_model(grammar, corpus, epochs=30, device=get_device())

print(model.generate_sentence(seed_noise=1.0))   # 文法约束生成
h = model.encode_sentence("the cat sees a dog")  # 解析 + 编码
print(model.generate_sentence(root_embedding=h)) # 从向量解码
```

---

## 演示结果

以下为 `python -m rvnn_text.demo` 在 Apple Silicon（MPS）上的实测输出（1500 句、20 epochs、`dim=64`）。

### 1. 训练收敛

| epoch | loss   | rule    | word    | recon   | val     |
|-------|--------|---------|---------|---------|---------|
| 1     | 0.3153 | 0.0421  | 0.0950  | 0.1783  | 0.1491  |
| 5     | 0.0887 | 0.0003  | 0.0008  | 0.0876  | 0.0863  |
| 10    | 0.0741 | 0.0000  | 0.0002  | 0.0739  | 0.0736  |
| 15    | 0.0676 | 0.0000  | 0.0001  | 0.0675  | 0.0672  |
| 20    | 0.0603 | 0.0000  | 0.0000  | 0.0602  | 0.0604  |

规则损失（rule）在第 9 个 epoch 后降至 0，单词损失（word）在第 16 个 epoch 后归零——解码器已完全掌握文法；剩余损失主要来自重构项（recon）。

### 2. 文法内化：held-out 准确率

```
held-out rule-prediction accuracy: 100.0%
held-out word-prediction accuracy: 100.0%
```

### 3. 文法约束生成（采样）

```
Bob sees
Alice likes quickly
Mary likes a happy clever small cat
a red tree chases quickly
```

递归规则 `NOM → Adj NOM` 让模型能生成任意长度的形容词链，且永远合乎文法：

```
S
├── NP
│   └── Proper: Mary
└── VP
    ├── Verb: likes
    └── NP
        ├── Det: a
        └── NOM
            ├── Adj: happy
            └── NOM
                ├── Adj: clever
                └── NOM
                    ├── Adj: small
                    └── NOM
                        └── Noun: cat
```

### 4. 编码-解码往返（round-trip）

句子解析成树 → 编码为根向量 → 从根向量重新解码，可逐词重建：

```
input:  the happy cat sees a dog
output: the happy cat sees a dog

input:  every clever girl likes the robot
output: every clever girl likes the robot
```

---

## 项目结构

```
rvnn-text/
├── rvnn_text/
│   ├── grammar.py      # 上下文无关文法、句法树、采样 / 解析 / 树可视化
│   ├── model.py        # RvNN 编码器 + 文法约束递归解码器、训练目标、评估
│   ├── data.py         # 从文法生成语料、train/val 划分
│   ├── train.py        # 训练循环（fire CLI）
│   ├── generate.py     # 生成（fire CLI）
│   ├── demo.py         # 端到端演示（fire CLI）
│   ├── checkpoint.py   # 模型保存 / 加载
│   └── utils.py        # 设备选择、随机种子
├── tests/              # pytest 测试（文法、模型、端到端）
├── pyproject.toml
└── requirements.txt
```

运行测试：

```bash
python -m pytest
```

---

## 数据来源

本项目为了**零依赖、开箱即用**，用内置文法 `Grammar` 自行生成合成语料。若要换用真实数据（尤其是带
句法标注的），可参考以下来源（SST 与 RvNN 主题最契合）：

| 来源 | 地址 | 说明 |
|---|---|---|
| UCI Machine Learning Repository | <https://archive.ics.uci.edu/> | 经典机器学习数据集 |
| Kaggle | <https://www.kaggle.com/> | 竞赛与社区数据集 |
| IEEE DataPort | <https://ieee-dataport.org/> | 科研数据集（IEEE 官方） |
| Stanford Sentiment Treebank | <https://nlp.stanford.edu/sentiment/> | 成分句法树 + 逐节点情感标签，RvNN 经典基准 |

接入真实数据的方法：把外部语料的成分句法树转换成 `rvnn_text.grammar.Node`（叶子为词性 + 单词，
内部节点为短语类别），再复用 `train_model` 即可。

---

## 扩展方向

- **变分先验（VAE）**：当前的生成先验是"可学习均值向量 + 高斯噪声"，解码会偏向某个区域。
  改为变分自编码器（对根向量学习均值/方差并加 KL 正则）可得到平滑、可控的生成分布。
- **真实数据**：在 SST 上训练，做短语级情感分析。
- **递归批处理**：当前按树逐个训练；对同构子树做批处理（如 TreeLSTM 的批处理技巧）可显著加速。
- **更丰富的组合函数**：如 Recursive Neural Tensor Network（Socher 2013）、Gated RvNN（TreeLSTM）。
- **任意文法**：`Grammar` 支持自定义产生式（arity ≤ 2），可直接换成中文、代码或 SQL 的文法。

---

## License

MIT
