Restaurant Review Ratings Prediction
## Restaurant Review Ratings Prediction

<a href="https://www.linkedin.com/in/mohit-matta-61b65b18"> Author: Mohit Matta </a>

<a href="https://mohitmatta.github.io/">Click here to go back to Portfolio Website </a>

![A remote image](https://github.com/mohitmatta/restaurant_review_rating_prediction/blob/9e7a1ab704cf93754d491b9486f45ac33bba30d5/results/images.jpeg)


## Abstract: 

In recent years, abundant online review data has become available for analysis through different websites connecting customers and businesses. Many websites allow users to post online reviews and share their experiences about the food restaurants. The wealth of information hidden behind the online review data can help the restaurant business and startups to set up defining goals to improve the revenue. A large amount of plain text data that customers have posted can help on sustainability and companies can come up with a sustainable marketing strategy to understand and meet diverse customer demands and maintain competitive advantages. This case study can help restaurants to determine how well they are performing as well as improve customer satisfaction.


### Problem Statement

Users also try to convey hidden emotions and linguist styles through textual review comments that are often ignored since the performance is always measured in numerical factors like ratings. Sometimes the relationship between ratings and reviews is not obvious and users just look at the rating and decide to go ahead with the restaurants.
Review rating prediction has great importance because the users can decide whether to look at the textual reviews or ignore them. The review comment given by two different users can lead to different ratings because the reason for the online reviews for the users will be different. The online data related to reviews is increasing at a very high rate. Yelp recommendation website data has increased with an annual rate of 9% from 2009 to 2020 with a total of around 224 million reviews. This project will not only inform what factors matter most for the customers but also tells what features the new restaurant should focus on in order to successfully run it.

### Methods used in my project

Data was processed, cleaned and transformed using python scripts. After separating the restaurant businesses from rest, when data was explored it was found that reviews text field is in free form text format.Customers use different cases , grammatically incorrect words , punctuation marks and slang words in the review. They may use different words for expressing their likes or dislikes.Stop words are used very frequently by users which are not needed for machine learning models for analysis.So standard natural language processing techniques were used to convert all cases to lowercase , to remove punctuation, to remove stop words and apply porter stemmer to convert the sentences to stemmed sentences.

We used different feature extraction methods based on semantic analysis to extract useful features from the review corpus and built a feature vector for each review. Both unigram and bigram methods were used for the study. We used count vectorizer objects, word level TF-IDF(Term Frequency-Inverse Document Frequency) objects , n-gram level vectorizer objects and Character level vectorizer objects for vector representation of review text. Count vectorizer and TF-IDf vectorizer create a dictionary of words from review dataset and consider each unique word as a feature.Once the frequency of each word is obtained TFIDF weighing technique creates the final feature matrix.It adds high weights to words that are rarely occurring in the text and less weight to words that are frequently occurring. Bigram models gives better results compared to count vectorizers because it considers the relationship between two words. So when TFIDF weighing technique is applied on the text , more importance is given to combination of words like ‘delicious food’ and ‘coming back’ compared to single and more common occurrences.
Supervised Learning
To train our prediction models, we use four supervised learning algorithms.

Logistic Regression 
The logistic regression model predicts the conditional probability using the feature vectors and decides the class labels. The results that give the highest probability are mapped to the output as the final rating for the review.

Naive Bayes Classification
A Naive Bayes classification works on the basis of the Naive Bayes theorem of independent assumption between the features. It also works on conditional probability and constructs a classifier based on the probability model.

Linear Support Vector Classification (SVC) 
In the SVM technique, we select the best hyperplane (decision boundary) from the data points and separates the labels. We are using linear SVM techniques where data is linearly separable. It is also known as a discriminative classifier. It also uses the Kernel trick for non-linear classification.

Gradient Boost Classification
Gradient Boost classification uses ensemble techniques which works on the principle that a collection of predictors(weak or strong) works better than individual predictors. In the Boost technique, the weak learners are converted into strong learners. The model tries to minimize the overall error of the strong learners.
Neural Network Classification
A neural network has the weight, score function, and loss function as the main components. It learns in a feedback loop. It adjusts the weight based on results from score function and loss function. The architecture of a neural network has an input layer, hidden layer, and output layers. The neural network calculates, tests, calculates again, tests again, and repeats until the optimum and accurate solution is reached.

Word2Vec Gradient Boost Classification              
Word2vec works on the idea of distributional semantics which means that we can understand the meaning of a word by understanding the context that a word keeps. It was developed to overcome the shortcomings of one-hot encoding. With the increase in word's vector dimensions,  the relationship between words can be explained in detail.For example, if we have words like apple, mango, and element word2vec model will create features like is_fruit, is_eatable and is_animal. Fruits apple and mango will have the same context for is_fruit and is_animal but less contextual similarity for is_animal.

Accuracy, precision, and recall statistics were produced to determine the overall performance of each model.  



### Project Dataset:

##yelp_business.json
- Type:		JSON
- Columns: 	13
- Rows:		11537

##yelp_reviews.json
- Type:		JSON
- Columns:  8
- Rows:	229907

##yelp_users.json
- Type:		JSON
- Columns:  6
- Rows:	43873
- 
## Included Project Variables / Factors 

 | Feature / Factors | Definition | Type |
 | --------- | --------- | ---------- |
|business_id| 22 character unique string business id| string|
|name| the business's name| string|
|address| the full address of the business|string|
|city|the city | string|
|state| 2 character state code| string|
|postal code |the postal code| string|
|latitude | latitude|float|
|longitude | longitude|float|
|stars | star rating, rounded to half-stars |float|
|review_count | number of reviews|integer|
|is_open |  0 or 1 for closed or open, respectively| integer|
|attributes | business attributes to values | object|
|categories | an array of strings of business categories | object|
|hours | an object of key day to value hours, hours are using a 24hr clock| object|



| Feature / Factors | Definition | Type |
 | --------- | --------- | ---------- |
|review_id| 22 character unique review id| string|
|user_id| 22 character unique user id, maps to the user in user.json| string|
|business_id| 22 character business id, maps to business in business.json|string|
|stars|star rating | integer|
|date|  date formatted YYYY-MM-DD| string|
|text |the review itself| string|
|useful |  number of useful votes received|integer|
|funny | number of funny votes received|integer|
|cool | number of cool votes received |integer|



 

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
| restaurant_review_rating_prediction.py | EDA/Model building |



## Datasets
| File  | Description |
| ------ | ------ |
| yelp_training_set_business.json | Kaggle Link- https://www.kaggle.com/c/yelp-recsys-2013/data#:~:text=get_app-,Download,-All |
| yelp_training_set_review.json | Kaggle Link- https://www.kaggle.com/c/yelp-recsys-2013/data#:~:text=get_app-,Download,-All | 



## Sequence of programs for execution


1) Download tha Yelp review , Phoenix,AZ dataset from Kaggle Link  https://www.kaggle.com/c/yelp-recsys-2013/data#:~:text=get_app-,Download,-All 
2) Run restaurant_review_rating_prediction.py that will perform EDA,build and execute the Models



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


[1] M. Anderson and J. Magruder. Learning from the crowd. 2011. 

[2] E. Cambria, B. Schuller, Y. Xia, and C. Havasi. New avenues in opinion mining and sentiment analysis. IEEE Intelligent Systems, (2):15–21, 2013. 

[3] M. Fan and M. Khademi. Predicting a business star in yelp from its reviews text alone. arXiv:1401.0864, 2014. [4] Y. Koren. Factorization meets the neighborhood: a multifaceted collaborative filtering model. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 426–434. ACM, 2008. 

[5] A. Mueller. A wordcloud in python. Andy’s Computer Vision and Machine Learning Blog, June 2012.

[6] A. Ortony, G. L. Clore, and A. Collins. The cognitive structure of emotions. Cambridge university press, 1990. [7] R. A. Stevenson, J. A. Mikels, and T. W. James. Characterization of the affective norms for english words by discrete emotional categories. Behavior Research Methods, 39(4):1020–1024, 2007. 

[8] M. Woolf. The statistical difference between 1-star and 5-star reviews on yelp. http://minimaxir.com/2014/09/one-star-five-stars/, September 2014.

[9] Narendra Gupta, Giuseppe Di Fabbrizio, and Patrick Haffner. 2010. Capturing the stars: predicting ratings for service and product reviews. In Proceedings of the NAACL HLT 2010 Workshop on Semantic Search, pages 36–43. Association for Computational Linguistics.

[10]López Barbosa, R.R., Sánchez-Alonso, S. and Sicilia-Urban, M.A. (2015), "Evaluating hotels rating prediction based on sentiment analysis services", Aslib Journal of Information Management, Vol. 67 No. 4, pp. 392-407. https://doi.org/10.1108/AJIM-01-2015-0004
 


:bowtie:
