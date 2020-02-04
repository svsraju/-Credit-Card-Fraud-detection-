# Credit Card Fraud Detection Using Support-Vector-Machines
Exploring support vector machines (SVMs) using GridsearchCV and working with highly unbalanced dataset: European Cardholder Credit Card Transactions

![alt_text](https://lh4.googleusercontent.com/proxy/1b4sPrViWc4PFplKgTrIDcYGiIo94tlcXeOYXlY5X788TL3cwiSPDn9yfv9SAsd5DlXSEJViK84IsXfr3-44iXyoFq3Ozw)


### Project
For this project I will be exploring support vector machines (SVMs)
using GridsearchCV and working with highly unbalanced datasets.


### [Data set](https://www.kaggle.com/kerneler/starter-credit-card-fraud-detection-e6d0de2d-9)
European Cardholder Credit Card Transactions, September 2013  
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

Confusion Matrix  
[TN  FP]  
[FN  TP]

Accuracy = $\frac{TN + TP}{TN + TP + FN + FP}$  
TPR = $\frac{TP}{TP + FN}$  
FPR = $\frac{FP}{FP + TN}$  

Recall = TPR = $\frac{TP}{TP + FN}$  
Precision = $\frac{TP}{TP + FP}$  
F1 Score = 2 * $\frac{precision * recall}{precision + recall}$  

See the general resoucres below for more details on precision, recall, and the F1 score.


The dataset was collected and analysed during a research collaboration of 
Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université 
Libre de Bruxelles) on big data mining and fraud detection [1]

[1] Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.
Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium
on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.
http://mlg.ulb.ac.be/BruFence . http://mlg.ulb.ac.be/ARTML

### Please check the notebook for complete analysis

### Results
From our test results for our best model, I get a F1 score of 0.7273. As F1 score summarizes both precision and recall we can see that our model can perform better as the value is not close to 1, If we would have got a F1 score of 1 we can conclude our model was predicting all the values correct.But, In our case since we have very less positive values it was difficult to balance the data.That must be reason for having a low F1 score

### Difference in the meaning of the AUC for the ROC vs the AUC for the PRC.
Since, we are dealing with class imbalance problem, using acccuracy or any other metric might not give us resonable results.so we consider ROC AUC(Receiver Operating Characteristic area under curve) and PRC AUC(Precision Recall area under curve).

A ROC curve is plotting True Positive Rate against False Positive Rate, we would like to have a model to be at the upper left corner which in other words is to have a model which will not give us any false positives.ROC AUC is the area under the ROC curve. The higher it is, the better the model.For our best model we have 0.944
for ROC AUC, which is pretty good

A PR curve is plotting Precision against Recall, we would like to have a model be at the upper right corner, which is basically getting only the true positives with no false positives and no false negatives.The PRC AUC is just the area under the PRC curve. The higher it is, the better the model.For our best model we have 0.522 for ROC AUC, which is not really good.

Since our problem has a more negatives than positives and PRC does not consider true negatives it would be better to use PRC as our metric. Using ROC as a metric we might end up thinking our model is performing really well, but reality is it considers True Negatives and if our data has lots of negative examples we will end up getting better results for ROC AUC, which I believe the case above.I have a really good ROC AUC value, but my PRC AUC is not close to good.I can say that my model is not performing well on this data, we might want to check other classifiers to check their performance.


### General References
* [Guide to Jupyter](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)
* [Python Built-in Functions](https://docs.python.org/3/library/functions.html)
* [Python Data Structures](https://docs.python.org/3/tutorial/datastructures.html)
* [Numpy Reference](https://docs.scipy.org/doc/numpy/reference/index.html)
* [Numpy Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf)
* [Summary of matplotlib](https://matplotlib.org/3.1.1/api/pyplot_summary.html)
* [DataCamp: Matplotlib](https://www.datacamp.com/community/tutorials/matplotlib-tutorial-python?utm_source=adwords_ppc&utm_campaignid=1565261270&utm_adgroupid=67750485268&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=1t1&utm_creative=332661264365&utm_targetid=aud-299261629574:dsa-473406587955&utm_loc_interest_ms=&utm_loc_physical_ms=9026223&gclid=CjwKCAjw_uDsBRAMEiwAaFiHa8xhgCsO9wVcuZPGjAyVGTitb_-fxYtkBLkQ4E_GjSCZFVCqYCGkphoCjucQAvD_BwE)
* [Pandas DataFrames](https://urldefense.proofpoint.com/v2/url?u=https-3A__pandas.pydata.org_pandas-2Ddocs_stable_reference_api_pandas.DataFrame.html&d=DwMD-g&c=qKdtBuuu6dQK9MsRUVJ2DPXW6oayO8fu4TfEHS8sGNk&r=9ngmsG8rSmDSS-O0b_V0gP-nN_33Vr52qbY3KXuDY5k&m=mcOOc8D0knaNNmmnTEo_F_WmT4j6_nUSL_yoPmGlLWQ&s=h7hQjqucR7tZyfZXxnoy3iitIr32YlrqiFyPATkW3lw&e=)
* [Sci-kit Learn Linear Models](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)
* [Sci-kit Learn Ensemble Models](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)
* [Sci-kit Learn Metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
* [Sci-kit Learn Model Selection](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection)
* [Scoring Parameter](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
* [Scoring](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring)
* [Plot ROC](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html)
* [Precision, Recall, F1 Score](https://en.wikipedia.org/wiki/Precision_and_recall)
* [Precision-Recall Curve](https://acutecaretesting.org/en/articles/precision-recall-curves-what-are-they-and-how-are-they-used)
* [Probability Plot](https://www.itl.nist.gov/div898/handbook/eda/section3/normprpl.htm)
