# Credit Card Fraud Detection Using Support-Vector-Machines
Exploring support vector machines (SVMs) using GridsearchCV and working with highly unbalanced dataset: European Cardholder Credit Card Transactions

![image](https://user-images.githubusercontent.com/46058709/82122058-d123fe00-9756-11ea-9c6c-97471315980d.png)

## Motivation

It is very important to detect any Fraud transcations and notify the user the same. Our idea was to develop a Fraud detection application that financial institutions can use to assess account behavior in each operation and can make a real-time judgment on whether a transaction is fraudulent. For this project I will be exploring support vector machines (SVMs)
using GridsearchCV and working with highly unbalanced datasets.


### [Data set](https://www.kaggle.com/kerneler/starter-credit-card-fraud-detection-e6d0de2d-9)
I have used European Cardholder Credit Card Transactions, September 2013  
This dataset presents transactions that occurred over two days. There were 492 incidents of 
frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class 
(frauds) accounts for 0.197% of all transactions.

__Features__  
* V1, V2, ... V28: are principal components obtained with PCA  
* Time: the seconds elapsed between each transaction and the first transaction  
* Amount: is the transaction Amount  
* Class: the predicted variable; 1 in case of fraud and 0 otherwise.  

Given the class imbalance ratio, it is recommended to use precision, recall and the 
Area Under the Precision-Recall Curve (AUPRC) to evaluate skill. Traditional accuracy 
and AUC are not meaningful for highly unbalanced classification. These scores are 
misleading due to the high impact of the large number of negative cases that can easily
be identified. 

Examining precision and recall is more informative as these disregard the number of 
correctly identified negative cases (i.e. TN) and focus on the number of correctly 
identified positive cases (TP) and mis-identified negative cases (FP). Another useful 
metric is the F1 score which is the harmonic mean of the precision and recall; 1 is the 
best F1 score.

The dataset was collected and analysed during a research collaboration of 
Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université 
Libre de Bruxelles) on big data mining and fraud detection [1]

[1] Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.
Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium
on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.
http://mlg.ulb.ac.be/BruFence . http://mlg.ulb.ac.be/ARTML

### Please check the notebook for complete analysis

### Results
- From our test results for our best model, I get a F1 score of 0.7273. 
- As F1 score summarizes both precision and recall we can see that our model can perform better as the value is not close to 1, If we would have got a F1 score of 1 we can conclude our model was predicting all the values correct.
- But, In our case since we have very less positive values it was difficult to balance the data.That must be reason for having a low F1 score

### Understanding the -Difference in the meaning of the AUC for the ROC vs the AUC for the PRC.
- Since, we are dealing with class imbalance problem, using acccuracy or any other metric might not give us resonable results.so we consider ROC AUC(Receiver Operating Characteristic area under curve) and PRC AUC(Precision Recall area under curve).

- A ROC curve is plotting True Positive Rate against False Positive Rate, we would like to have a model to be at the upper left corner which in other words is to have a model which will not give us any false positives.
- ROC AUC is the area under the ROC curve. The higher it is, the better the model.For our best model we have 0.944
for ROC AUC, which is pretty good

- A PR curve is plotting Precision against Recall, we would like to have a model be at the upper right corner, which is basically getting only the true positives with no false positives and no false negatives.The PRC AUC is just the area under the PRC curve. The higher it is, the better the model.For our best model we have 0.522 for ROC AUC, which is not really good.

- Since our problem has a more negatives than positives and PRC does not consider true negatives it would be better to use PRC as our metric. Using ROC as a metric we might end up thinking our model is performing really well, but reality is it considers True Negatives and if our data has lots of negative examples we will end up getting better results for ROC AUC, which I believe the case above.I have a really good ROC AUC value, but my PRC AUC is not close to good.

