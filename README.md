# Kaggle Google Quest Challenge

## Competition URL
[竞赛网址](https://www.kaggle.com/competitions/google-quest-challenge/overview)

## 写在前面：第一次从零开始完成一个深度学习项目，比起coursera和李宏毅教授的补齐代码作业相比，感觉麻烦了不少，项目好像也不算简单。

## 1. 数据介绍  
- 1.1: 输入：
   - question_type
   - question_body
   - question_body
   - question_user_page
   - answer
   - url
   - question_user_name
   - question_user_name等  
- 1.2: 输出：30项label，都介于[0,1]，都是衡量question和answer的指标，如：
   - question_body_critical
   - question_not_really_a_question
   - answer_type_instructions等

## 2. 特征工程  
- 2.1. 先查看各项输入的长度，输入的词云，各项输出的分布，参考：
   - [Kaggle上的一条帖子](https://www.kaggle.com/code/manikanthgoud/google-quest-challenge-data-preprocessing-fe)。
   - 有以下结果：
     - 2.1.1：输入中question_body，answer长度90%都在280词以下，但各自都有一些1000词以上的data，question_type长度80%在60词以下，其余项一般不长
     - 2.1.2：输出全部为classification型，但有几项label存在不均衡问题  
- 2.2. 特征的增删改：基本思考是，由于label都是衡量question和answer的指标，大部分question_body+answer+question_type的长度刚好就是512左右，很难不让人浮想联翩。但尝试别的方面：
   - 2.2.1：有没有可能website或者user和question、answer有一定的关系？
   - 2.2.2：url和question_type_spelling的关系，发现：
     - url包含ell.stackexchange.com与english.stackexchange.com的项，question_type_spelling为0.5，否则为0
   - 2.2.3：去缩写：如won't -> will not
   - 2.2.4: 根据[stopwords](https://gist.github.com/sebleier/554280)来分词  
   - 2.2.5：词根还原  
   - 2.2.6：拼写错误纠正
   - 2.2.7：关于字符串截断：一般过长的字符串截中间留首尾

## 3. 模型选择  
我的理想情况下是这样的：
- 3.1：分成Q、A和QA三组，Q组模型包含question_type+question_body，回答只关于question的label。
- A组包含answer，回答只包含answer的label。
- QA组模型包含三项question_type+question_body+A组包含answer，回答question与answer匹配度的问题。
- 每组对应的label提前人工分好类，并且删去与其他feature有强相关的label。
- 3.2：每组包含Roberta，XLNet，Bert三个模型，每个模型取隐藏层[-1, -3, -5, -7, -9]的输出，拼接，用一个全连接+selu/relu/tanh展平，再用一个全连接+softmax预测30个label，dropout（0.3）  
- 3.3：三个模型预测结果相近取平均，三者不相近则取最相近的两者的平均
- 3.4：新学的技巧：
    ```
    optimizer = Adafactor(net.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    lr_scheduler = AdafactorSchedule(optimizer)
    ```
    Adafactor更适合transformer结构，而且可以自动调参。

    考虑到计算资源限制，最终采用了单模型RobertaForSequenceClassification。

## 4. 一点总结
- 4.1：可以尝试写出更复杂的分类head，更多样的模型来平均
- 4.2：特征的处理真的很重要，增删改都要慎重，好的数据比花几个小时调参有用
- 4.3：记得attention_mask和[token,segment,position]来embed


## 5. 更新
- 5.1：使用自定义的分类器，对roborta的最后4层layer的output（size=4*768）用三层全连接分类，各种活化函数的效果提升不大
