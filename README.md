[竞赛网址](https://www.kaggle.com/competitions/google-quest-challenge/overview)
1.数据介绍  
 *  1.1:输入：question_type,question_body,question_body,question_user_page,answer,url,question_user_name,question_user_name等  
 *  1.2:输出：30项label，都介于[0,1]，都是衡量question和answer的指标，如question_body_critical，question_not_really_a_question，answer_type_instructions等  
2.特征工程  
  2.1.先查看各项输入的长度，输入的词云，各项输出的分布，这一点可以参考[kaggle上的一条帖子](https://www.kaggle.com/code/manikanthgoud/google-quest-challenge-data-preprocessing-fe)。有以下结果：  
  
    2.1.1：输入中question_body，answer长度90%都在280词以下，但各自都有一些1000词以上的data，question_type长度80%在60词以下，同样有长句，其余项一般不长  
    2.1.2：输出全部为classification型，但有几项label存在不均衡问题  
    
  2.2.特征的增删改：基本的思考是，由于label都是衡量question和answer的指标，而且大部分question_body+answer+question_type的长度刚好就是512左右，很难不让人浮想联翩，有的高分code就是直接取这三项。但我觉得还是应该试试别的一些方面：  
  
    2.2.1：有没有可能website或者user和question、answer有一定的关系？同一个user提供的answer或者problem质量应该是类似的；有的劣质网站可能普遍answer或者problem质量差一点。但由于重复的user不多，所以没深入考虑  
    2.2.2：url？url和question_type_spelling有关系！发现：有url包含ell.stackexchange.com与english.stackexchange.com的项，question_type_spelling为0.5，否则为0  
    
    ok，考虑完其他的特征，现在考虑主要的这三项：question_type，question_body，answer，先是要清洗一下，这里给几篇kaggle上面总结的References for feature engineering:  
    [1](https://www.kaggle.com/c/google-quest-challenge/discussion/130041) - meta features.  
    [2](https://www.kaggle.com/codename007/start-from-here-quest-complete-eda-fe?scriptVersionId=25618132&cellId=65) - tfidf, count based features  
    [3](https://towardsdatascience.com/hands-on-transformers-kaggle-google-quest-q-a-labeling-affd3dad7bcb) - web scraping features  
    包括：  
    2.2.3：去缩写：如won't -> will not  
    2.2.4: 根据[stopwords](https://gist.github.com/sebleier/554280)来分词  
    2.2.5：词根还原  
    2.2.6：拼写错误纠正  

3.模型选择：我的理想情况下是这样的：  
  3.1：分成Q、A和QA三组，Q组模型包含question_type+question_body，回答只关于question的label。A组包含answer，回答只包含answer的label。QA组模型包含三项question_type+question_body+A组包含answer，回答question与answer匹配度的问题。当然每一组对应的label提前人工分好类，并且删去与其他feature有强相关的label。  
  3.2：每组包含Roberta，XLNet，Bert三个模型，每个模型取隐藏层[-1, -3, -5, -7, -9]的输出，拼接，用一个全连接+selu/relu/tanh展平，再用一个全连接+softmax预测30个label，dropout（0.3）  
  3.3：三个模型预测结果相近取平均，三者不相近则取最相近的两者的平均  
  
  
