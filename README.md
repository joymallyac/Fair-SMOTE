# Fair-SMOTE
This repo is created for FSE 2021 paper - **Bias in Machine Learning Software: Why? How? What to do?**

Increasingly, software is making autonomous decisions in case of criminal sentencing, approving credit cards, hiring employees, and so on. Some of these decisions show bias and adversely affect certain social groups (e.g. those defined by sex, race, age, marital status). Many prior works on bias mitigation take the following form: change the data or learners in multiple ways, then see if any of that improves fairness. Perhaps a better approach is to postulate root causes of bias and then applying some resolution strategy.


This paper postulates that the root causes of bias are the prior decisions that affect- (a) what data was selected and (b) the labels assigned to those examples. Our Fair-SMOTE algorithm removes biased labels; and rebalances internal distributions such that based on sensitive attribute,  examples are equal in both positive and negative classes. On testing, it was seen that this method was just as effective at reducing bias as prior approaches. Further, models generated via Fair-SMOTE achieve higher performance (measured in terms of recall and F1) than other state-of-the-art fairness improvement algorithms.

To the best of our knowledge, measured in terms of number of analyzed learners and datasets,
this study is one of the largest studies on bias mitigation yet presented in the literature.

## Dataset Description - 

1> Adult Income dataset - http://archive.ics.uci.edu/ml/datasets/Adult

2> COMPAS - https://github.com/propublica/compas-analysis

3> German Credit - https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29 

4> Bank Marketing - https://archive.ics.uci.edu/ml/datasets/bank+marketing

5> Default Credit - https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

6> Heart - https://archive.ics.uci.edu/ml/datasets/Heart+Disease

7> MEPS - https://meps.ahrq.gov/mepsweb/

8> Student - https://archive.ics.uci.edu/ml/datasets/Student+Performance

9> Home Credit - https://www.kaggle.com/c/home-credit-default-risk

## Code Description -

* At first, download csv files from the above links and copy them in the `data` folder.
* `Data_Visualization.ipynb` (inside `Data_Visualization`) file shows the data imbalance present in all the datasets
* `Root_Cause` folder contains files for two main reasons behind training data bias
  * `Data_Imbalance_Problem.ipynb` shows problems of traditional class balancing methods
  * `Label_Problem.ipynb` shows problems of label bias
* `SMOTE.py` is SMOTE (Synthetic Minority Over-sampling Technique) class balancing method
* `Generate_Samples.py` contains code for generating samples based on Fair-SMOTE
* `Measure.py` contains python definition of all the fairness and performance metrics
* `Statistical Test` contains `Stats.py` which has `ScottKnott` implementation
* `Fair-SMOTE` and `Fair_Situation_Testing` contains of examples of how to use `Fair-SMOTE` and `Fair Situation Testing`
* For reproducing results of the paper, please run `RQ2.sh`, `RQ3.sh` and `RQ5.sh`

## Data Preprocessing -
* We have used data preprocessing as suggested by [IBM AIF360](https://github.com/IBM/AIF360)
* The rows containing missing values are ignored, continuous features are converted to categorical (e.g., age<25: young,age>=25: old), non-numerical features are converted to numerical(e.g., male: 1, female: 0). Fiinally, all the feature values are normalized(converted between 0 to 1). 
* For `optimized Pre-processing`, plaese visit [Optimized Preprocessing](https://github.com/Trusted-AI/AIF360/blob/master/aif360/algorithms/preprocessing/optim_preproc.py)
