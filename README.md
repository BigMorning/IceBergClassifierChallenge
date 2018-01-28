# IceBergClassifierChallenge

Kaggle Statoil/C-CORE Iceberg Classifier Challenge 比赛
public：40/3343, private: 46/3343
EDAbegin：前期做的一些EDA工作，主要包括数据探索，2D图片展示，3D图片展示。
MyModel：包含所训练的三个CNN模型（Vgg16, Res50和自己搭建的CNN），其中表现最好的是Vgg16，public能到0.1463，Res50也能到0.1468，自己搭的CNN在0.16左右。
Kernel：包含三个我选取的Kernel中结果最好的模型
- 1 Ensembling GBMs（Chia-Ta）https://www.kaggle.com/cttsai/ensembling-gbms-lb-203
- 2 VggBn(monkeyking) https://www.kaggle.com/supersp1234/best-single-model-lb-0-1400
- 3 Keras CNN with Pseudolabeling(TheGruffalo) https://www.kaggle.com/cbryant/keras-cnn-with-pseudolabeling-0-1514-lb
- 4 PyTorch CNN DenseNet Ensemble(QuantScientist) https://www.kaggle.com/solomonk/pytorch-cnn-densenet-ensemble-lb-0-1538(未加入)

我的最好的Blend结果包括我自己训练的3个模型，Vgg16，Res50，CNN和Kernel中选取的GBMs，VggBN，PseudolabelingCNN，DenseNet，还有我之前提交的LB最好的结果。Blend的策略很简单，类似于一个打分机制，给LB最好的结果5分，mean、median等特征3分，其它模型给1分，最后得分相加后归一化。然后再对不同分数段的结果分别给不同的预测值，比如>0.85的，给所有结果的最大值，小于0.15的给最小值，之间的给一个中和的结果。



