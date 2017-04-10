# Introduction
This project is implementation of **Probabilistic Generative Model** and **Probabilistic Discriminative Model** for multi-class classification. (*see Pattern Recognition and Machine Learning, Bishop 2006*)
Classifcation task can be splitted into two stages - inference and decision.  Probabilistic Generative Model solve class posteriror via solving class conditional probabilities and class priors. 
Probabilistic Discriminative Model solve directly optimize linear combination weight with **Iterative Reweighted Least Squares (IRLS) - Newton-Raphson** to find class posteriror.  All datas are processed with **Principle Component Analysis (PCA)** or **Linear Discriminant Analysis (LDA)**.

# Dataset
Database of Faces ( AT&T Laboratories Cambridge)    
Reference : http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html

# Results
Best error rate of each model

|Probabilistic Generative Model   |  Probabilistic Discriminative Model |
|:-:|:-:|
|0.025   |  0.0 |

# Visualization (Decision boundary)
|Probabilistic Generative Model   |  Probabilistic Discriminative Model |
| ------------- |:------------:|
|![gen](/doc/bound_gen.png)|![dis](/doc/bound_dis_lda.png)|

# To train the model (examples)
Training scripts use default training data in data/class*.npy and default training hyperparameters. If you want to use your own data, please see the manual of main.py
```
./train_generative.sh {model output path}
./train_dicriminative.sh {model output path}
./train_dicriminative_lda.sh {model output path}
```

# To validate the model (examples)
```
./validate_generative.sh
./validate_dicriminative.sh 
./validate_dicriminative_lda.sh 
```

# To test the model (examples)
```
./test.sh {model input} {result output} {testing data} {model type [dis|gen]}

e.g.
./test.sh model/model-dis data/class1.npy,data/class2.npy,data/class3.npy dis
```

# To run demo
```
./demo.sh {model input} {model type [dis|gen]}

e.g.
./demo.sh model/model-dis dis

[04/11/2017 02:06:24 AM] Convert images at ./Demo to data/demo.npy
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [00:00<00:00, 76795.31it/s]
Demo images convertion done
[04/11/2017 02:06:24 AM] Load 600 data from ./data/demo.npy
[04/11/2017 02:06:24 AM] Loading stddev from model/model-dis_std.npy ...
[04/11/2017 02:06:24 AM] Loading basis from model/model-dis_basis.npy ...
[04/11/2017 02:06:24 AM] Loading model from model/model-dis.npy success [K = 3, M = 3]
[04/11/2017 02:06:24 AM] Use model dis with 3-dim (with bias) feautre space
[04/11/2017 02:06:24 AM] Converting to one-hot ...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [00:00<00:00, 908841.60it/s]
[04/11/2017 02:06:24 AM] Writing result to ./result/DemoTarget.csv ...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [00:00<00:00, 263848.02it/s]
```
