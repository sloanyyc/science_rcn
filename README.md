[![](data/vicarious_logo.png)](https://www.vicarious.com)

# Reference implementation of Recursive Cortical Network (RCN)

## Update
### 测试模式
需先训练后
```
python science_rcn/run.py --train_size 100 --test_size 20 --parallel --test_only
# 可以指定模型名称
python science_rcn/run.py --train_size 100 --test_size 20 --parallel --test_only --model_file='model.pkl'
```
### 指定数据集目录
参数 --data_dir=data/MNIST1

### 输出训练进度
输出当前训练的字符

### 输出测试进度，测试信息

```
python science_rcn/run.py --test_only
INFO:__main__:Testing on 20 images...
20:28:54
fwd_infer use 0.134



[ 0 17  1  3 11 12 23 18 21  4 15  2 16 24 14  6 19 10  5  7]
count 20
forward_pass use 0.854
try !! char: 0 win: 0 score: 24.4962846709
try !! char: 0 win: 3 score: 25.5222142712
try !! char: 0 win: 12 score: 26.7935296271
try !! char: 0 win: 23 score: 27.0901419072
try !! char: 0 win: 21 score: 27.0918954155
try !! char: 0 win: 4 score: 28.0573342511
try !! char: 0 win: 15 score: 28.205319167
try !! char: 0 win: 24 score: 29.0594048523
try !! char: 0 win: 19 score: 29.7853003534
try !! char: 0 win: 5 score: 30.2620724526
try !! char: 0 win: 7 score: 31.479678884
bwd_pass use 5.833
fwd_infer use 0.112
[27 26 17  8 12 11  3 21 18 15 23 28 14 24 10  6 19 16  7  5]
count 20
forward_pass use 0.871
try !! char: 0 win: 27 score: 33.8280306876
try !! char: 0 win: 26 score: 34.2232459437
...
```

## 一些问题
### 训练样本较多时，有时发生 IndexError 错误
捕捉错误，返回空，在外层将出错样本结果映射到第0个字符


Reference implementation of a two-level RCN model on MNIST classification. See the *Science* article "A generative vision model that trains with high data efficiency and breaks text-based CAPTCHAs" and [Vicarious Blog](https://www.vicarious.com/Common_Sense_Cortex_and_CAPTCHA.html) for details.

> Note: this is an unoptimized reference implementation and is not intended for production.

## Setup

Note: Python 2.7 is supported. The code was tested on OSX 10.11. It may work on other system platforms but not guaranteed.

Before starting please make sure gcc is installed (`brew install gcc`) and up to date in order to compile the various dependencies (particularly numpy).

Clone the repository:

```
git clone https://github.com/sloanyyc/science_rcn.git
```

Simple Install:

```
cd science_rcn
make
```

Manual Install (setting up a virtual environment beforehand is recommended):

```
cd science_rcn
python setup.py install
```

## Run

If you installed via `make` you need to activate the virtual environment:
```
source venv/bin/activate
```

To run a small unit test that trains and tests on 20 MNIST images using one CPU (takes ~2 minutes, accuracy is ~60%):
```
python science_rcn/run.py
```

To run a slightly more interesting experiment that trains on 100 images and tests on 20 MNIST images using multiple CPUs (takes <1 min using 7 CPUs, accuracy is ~90%):
```
python science_rcn/run.py --train_size 100 --test_size 20 --parallel
```

To test on the full 10k MNIST test set, training on 1000 examples (could take hours depending on the number of available CPUs, average accuracy is ~97.7+%):
```
python science_rcn/run.py --full_test_set --train_size 1000 --parallel --pool_shape 25 --perturb_factor 2.0
```

## Blog post

Check out our related [blog post](https://www.vicarious.com/Common_Sense_Cortex_and_CAPTCHA.html).

## MNIST licensing

Yann LeCun (Courant Institute, NYU) and Corinna Cortes (Google Labs, New York) hold the copyright of MNIST dataset, which is a derivative work from original NIST datasets. MNIST dataset is made available under the terms of the Creative Commons Attribution-Share Alike 3.0 license.
