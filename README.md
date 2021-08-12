Restaurant Review Ratings Prediction
## Restaurant Review Ratings Prediction

<a href="https://www.linkedin.com/in/mohit-matta-61b65b18"> Author: Mohit Matta </a>

<a href="https://mohitmatta.github.io/">Click here to go back to Portfolio Website </a>

![A remote image](https://github.com/mohitmatta/AML-Fraud-Transactions-Detection/blob/3144ff2db53fee08ce0af4f8149e4c8b7220a245/results/fraud_detection.jpeg)


## Abstract: 

With the recent rise in credit card usage as well as issuance of credit cards to customers, it has become increasingly important for financial institutions to curb the threat of fraudulent transactions and identity theft. Both modes of Credit cards usage(online and offline), although equally convenient, come with great risk.A little information about credit card(16 digit number) is required and sufficient to make a purchase , it can pose some risk of fraud as well. Fraud transaction detection systems can detect the transactions with great accuracy but most of them work after the transaction has occurred. It is important to analyze millions of transactions to identify the fraud pattern so that for future transactions we can detect it on spot and it can save millions of dollars to financial companies as well as customers.


### Problem Statement

Banks use a lot of traditional rule based systems to detect frauds but they create too many false positives that sometimes impact the customers adversely.Many of the global banks want to move away from traditional rule based engines towards predictive machine learning based monitoring systems for credit card transactions as well as other bank transfer operations involving money.The Global Digital Payment Market size is expected to reach $175.8 billion by 2026, rising at a market growth of 20% . As the transactions are supposed to be rising in future, banks need to equip themselves to better protect against frauds.Banks have a difficult task to look for a fraud transaction among millions of daily transactions.Due to rise and increase in amount of data, its becoming difficult day by day for humans to detect patterns arising from fraud transactions. Thats why machine learning techniques are becoming popular and are widely being used for fraud detection where information extraction is required from large datasets.


### Methods used in my project

Importing and basic cleaning and preprocessing of dataset was performed.The variables were examined to see the correlation.The data This is a classification problem to predict the fraud transactions, so we selected different classification models accordingly. With the objective being to provide a classification of fraud transactions, a predictive model was determined to be the best candidate. With the Exploratory Data Analysis, the data revealed that it was best to build advanced Data Science and Machine Learning models to identify and predict potentially suspicious transactions.With those two considerations, the models that were considered were Logistic Regression model, Random Forest,Gradient Boost model, Decision Tree and Neural deep learning algorithms.

### Project Dataset:

##FraudTrain.csv
- Type:		CSV
- Columns: 	23
- Rows:		1296675

##FraudTest.csv
- Type:		CSV
- Columns: 	23
- Rows:		555719

## Included Project Variables / Factors 

 | Feature / Factors | Definition | Type |
 | --------- | --------- | ---------- |
|row_id| record sequence| number|
|trans_date_trans_time| Timestamp of transaction| timestamp|
|cc_num| Credit Card Number|integer|
|Merchant| Merchant Name | Char|
|Category| Merchant Category | Char|
|amt |Transaction Amount| Decimal|
|First | First Name of customer|Char|
|last | Last Name of customer |Char|
|gender | Gender of customer |Char|
|street | Street address of customer|Char|
|city | City of customer| Char|
|state | State of customer | Char|
|Zip | Postal zip of customer | Int|
|Lat | Latitude coordinates of customer| Int|
|Long | Longitude coordinates of customer |Int|
|city_pop | city point of purchase- area in which marketers and retailers planned promotional activities surrounding the consumer products | Int|
|Job | Job name |Char|
|Dob | Date of Birth |Char|
|trans_num | Unique transaction Number|Int|
|unix_time | Unix format time of purchase|Int|
|mrch_lat | Merchant latitude location |Int|
|mrch_long | Merchant longitude location |Int| 
|is_fraud | 0 for non fraud transaction and 1 for fraud transaction|Int|

 
 

## Pythonic Libraries Used in this project
Package               Version
--------------------- ---------
- matplotlib            3.3.4 
- numpy                 1.20.1
- pandas                1.2.3
- scikit-learn          0.24.1 
- scipy                 1.6.1
- seaborn               0.11.1
- xgboost               1.5.0


## Repo Folder Structure

└───Datasets

└───Scripts

└───Presentation

└───Results

## Python Files 

| File Name  | Description |
| ------ | ------ |
| MOHIT_MATTA_DSC680_AML_Fraud_detection.py | EDA/Model building |



## Datasets
| File  | Description |
| ------ | ------ |
| fraudTrain.csv | Kaggle Link- https://www.kaggle.com/kartik2112/fraud-detection/download| 
| fraudTest.csv | Kaggle Link- https://www.kaggle.com/kartik2112/fraud-detection/download| 



## Sequence of programs for execution


1) Download tha Credit card transactions dataset from Kaggle Link  https://www.kaggle.com/kartik2112/fraud-detection/download
2) Run MOHIT_MATTA_DSC680_AML_Fraud_detection.py that will perform EDA,build and execute the Models



## Results


### EDA

### Transaction amount distribution for Fraud and Non Fraud Transaction types

![A remote image](https://github.com/mohitmatta/AML-Fraud-Transactions-Detection/blob/59759c5444c327b9931b2abdd9c0a7a920ad596e/results/EDA1.jpg)
![A remote image](https://github.com/mohitmatta/AML-Fraud-Transactions-Detection/blob/59759c5444c327b9931b2abdd9c0a7a920ad596e/results/EDA2.jpg)

### Transaction count of Fraud vs Non Fraud types for each category

![A remote image](https://github.com/mohitmatta/AML-Fraud-Transactions-Detection/blob/59759c5444c327b9931b2abdd9c0a7a920ad596e/results/EDA3.jpg)


### Top 20 locations with highest number of Fraud Transactions 
![A remote image](https://github.com/mohitmatta/AML-Fraud-Transactions-Detection/blob/59759c5444c327b9931b2abdd9c0a7a920ad596e/results/EDA10.jpg)


### Transaction count of Fraud vs Non Fraud types for each day of month
![A remote image](https://github.com/mohitmatta/AML-Fraud-Transactions-Detection/blob/59759c5444c327b9931b2abdd9c0a7a920ad596e/results/EDA5.jpg)

### Transaction count of Fraud vs Non Fraud types for each gender
![A remote image](https://github.com/mohitmatta/AML-Fraud-Transactions-Detection/blob/59759c5444c327b9931b2abdd9c0a7a920ad596e/results/EDA6.jpg)

### Transaction count of Fraud vs Non Fraud types for each month of year
![A remote image](https://github.com/mohitmatta/AML-Fraud-Transactions-Detection/blob/59759c5444c327b9931b2abdd9c0a7a920ad596e/results/EDA7.jpg)

### Transaction count of Fraud vs Non Fraud types for each day of week
![A remote image](https://github.com/mohitmatta/AML-Fraud-Transactions-Detection/blob/59759c5444c327b9931b2abdd9c0a7a920ad596e/results/EDA8.jpg)

### Top 20 job categories with most number of fraud transactions 
![A remote image](https://github.com/mohitmatta/AML-Fraud-Transactions-Detection/blob/59759c5444c327b9931b2abdd9c0a7a920ad596e/results/EDA9.jpg)

### Correlation Plot of diffrent variables
![A remote image](https://github.com/mohitmatta/AML-Fraud-Transactions-Detection/blob/59759c5444c327b9931b2abdd9c0a7a920ad596e/results/EDA11.jpg)

Based on correlation analysis performed in previous steps, most notable features selected for model input were listed below with ‘Is_fraud’ being target variable :
'amt', ‘hour','Age', 'category_food_dining', 'category_gas_transport','category_grocery_net', 'category_grocery_pos','category_health_fitness', 'category_home', 'category_kids_pets','category_misc_net', ‘category_misc_pos', 'category_personal_care','category_shopping_net', 'category_shopping_pos', 'category_travel', 'gender_M','gender_F','day_of_week_Monday','day_of_week_Tuesday', ‘day_of_week_Wednesday','day_of_week_Thursday','day_of_week_Saturday', ‘day_of_week_Sunday’,'city_pop','Cc_Nb_Tx_1Day_Window','Cc_Avg_Amount_1Day_Windo w','Cc_Nb_Tx_7Day_Window','Cc_Avg_Amount_7Day_Window','Cc_Nb_Tx_30Day_Window' ,'Cc_Avg_Amount_30Day_Window'



### Confusion Matrix & Model Comparison


![A remote image](https://github.com/mohitmatta/AML-Fraud-Transactions-Detection/blob/59759c5444c327b9931b2abdd9c0a7a920ad596e/results/Model1.jpg)
![A remote image](https://github.com/mohitmatta/AML-Fraud-Transactions-Detection/blob/59759c5444c327b9931b2abdd9c0a7a920ad596e/results/Model2.jpg)

### ROC Curve for Models

![A remote image](https://github.com/mohitmatta/AML-Fraud-Transactions-Detection/blob/59759c5444c327b9931b2abdd9c0a7a920ad596e/results/Model_ROC.jpg)

### Performance metrices of different models

![A remote image](https://github.com/mohitmatta/AML-Fraud-Transactions-Detection/blob/59759c5444c327b9931b2abdd9c0a7a920ad596e/results/Model_Comparison.jpg)

Compared against each other, Random Forest model is the best performing model with highest accuracy of 99.9%.




## References: 


1) Karimi ZandianZ., & KeyvanpourM. R. (2020). SSLBM: A New Fraud Detection Method Based on Semi- Supervised Learning. Computer and Knowledge Engineering, 2(2), 10-18. https://doi.org/10.22067/ cke.v2i2.82152
2) Ayano, O., & Akinola, S. O. (2017). A multi-algorithm data mining classification approach for bank fraudulent transactions. African Journal of Mathematics and Computer Science Research, 10(1), 5-13 .
3) Aleskerov, E., Freisleben, B., and Rao, B. CARDWATCH: A neural network based database mining system for credit card fraud detection. 1997. In Proceedings of IEEE/IAFE Conference on Computational Intelligence for Financial Engineering (New York City, NY, USA, March 23--25, 1997). CIFEr'97. IEEE, 220--226. DOI=http://dx.doi.org/10.1109/CIFER.1997.618940
4) Bansal, M. and Suman. Credit Card Fraud Detection Using Self Organised Map. 2014. International Journal of Information & Computation Technology. 4, 13 (2014), 1343–1348
5) J. Kingdon, "AI fights money laundering," in IEEE Intelligent Systems, vol. 19, no. 3, pp. 87-89, May- June 2004, doi: 10.1109/MIS.2004.1.
6) Kou Y, Lu C-TT, Sirwongwattana S, Huang YP, Sinvongwattana S (2004) Survey of fraud detection techniques. In: 2004 IEEE international conference on networking sensing and control, vol 2(3), pp 749– 754
7) Gao S, Xu D, Wang H, Green P (2009) Knowledge based anti money laundering: a software agent bank
application. J Knowl Manag 13(2):63–75
8) Chen Z, Van Khoa LD, Nazir A, Teoh EN, Karupiah EK (2014) Exploration of the effectiveness of
expectation maximization algorithm for suspicious transaction detection in anti-money laundering. ICOS
2014–2014 IEEE conference on open systems, pp 145–149
9) Bhattacharyya S, Jha S, Tharakunnel K, Westland JC (2011) Data mining for credit card fraud: a
comparative study. Decis Support Syst 50(3):602–613
10) Lv L-T, Ji N, Zhang J-L (2008) A RBF neural network model for anti-money laundering. In: Wavelet
 


:bowtie:
