# Introduction
This project is implementation of **Probabilistic Generative Model** and **Probabilistic Discriminative Model** for multi-class classification. (*see Pattern Recognition and Machine Learning, Bishop 2006*)
Classifcation task can be splitted into two stages - inference and decision.  Probabilistic Generative Model solve class posteriror via solving class conditional probabilities and class priors. 
Probabilistic Discriminative Model solve directly optimize linear combination weight to find class posteriror.

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
./validate_generative.sh {model output path}
./validate_dicriminative.sh {model output path}
./validate_dicriminative_lda.sh {model output path}
```
