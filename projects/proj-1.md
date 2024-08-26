---
layout: post
title: 'Project One'
---
_**Using machine learning to identify areas of interest in an anonymous airline survey**_

**Contents:**
1. Executive summary
  
   a. Objective

   b. Data source

   c. Rationale

   d. Resolution outline

   e. Conclusion

2. Data importing and preprocessing
  
   a. Data importing

   b. Cleaning

   c. Target/dependent variable splitting
   
3. EDA
  
   a. Basic descriptive statistics

   b. Outliers

   c. Categorical comparison

   d. Non-survey numerical against categorical

   e. Survey numerical against non-survey numerical

   f. A note on survey vs survey comparisons

   g. EDA conclusions
   
5. Methodology
  
   a. Train/validation/test split

   b. Initial model selection

   c. Hyperparameter testing
   -  i. RandomForestClassifier
   -  ii. SVC
   -  iii. XGBoost
   
   d. Cross-fold validation

   e. PCA and unsupervised learning

   f. Feature importance

   g. Results generation

9. Results and conclusion

   a. Results

   b. Conclusion

Executive Summary
---
**a. Objective**

The objective of this capstone project is to identify which features in a survey dataset most greatly affect 
the loyalty of a customer to a major commercial airline.

**b. Data Source**

The data was taken from an anonymised airline survey from Kaggle, (it has been 
suggested that this survey was taken by United, though this is not confirmed). The data 
was provided as a CSV.


Columns in this table are a mix of categorical and numerical data. Categorical features 
include _satisfaction_v2_; _Customer Type_; _Gender_; _Type of Travel_; and _Class_, 
representing customer satisfaction; brand loyalty; sex; travel purpose; and ticket class, 
respectively. For numerical data, there are _Flight Distance_, _Age_, _Departure Delay in 
Minutes_, _Arrival Delay in Minutes_ and several categorical “survey” columns taken 
from customers rating flight services from 0-5. There is also an additional I.D. column 
with integer data representing each individual customer, labelled _id_.


**c. Rationale**

Building a ML model capable of correctly classifying customer loyalty allows the airline 
to more effectively view which factors contribute to its long-term business success and 
growth of its brand. It may also thus help in reducing costs and identifying redundancies in 
the business model.


**d. Project outline**

1. Data importing and preprocessing – basic cleaning, identification of outliers.
2. EDA – basic statistics on both categorical and numerical columns. 
4. Further preprocessing – training; validation; and test data splitting.
5. Hyperparameter tuning – Model selection, k-fold cross-validation on test data. 
6. Results and conclusion.

 
**e. Conclusion**

I observed no major trends in age against distance for loyal customers, but there were 
significant repeated trends across numerous “classes” of customers who were disloyal 
to the brand. The airline can use this information to inform new business practice 
decisions. The two variables with the clearest patterns were _Age_ and _Flight Distance_.


Data importing and preprocessing
---

**a. Data importing**


The data was downloaded as a CSV from Kaggle[1]. It was then uploaded to a Jupyter 
Notebook file using Pandas’ read_csv() function.


**b. Cleaning**

I took steps to ensure the data I was using was fair and the model was able to learn 
from it. The first major step in doing this was to drop NaN values from the set. These 
removed only a few rows, and so I preferred deletion over replacement.


**c. Target/dependent variable splitting**

The target variable, y, was the _Customer Type_ column. The remaining 
columns were dependent variables.


EDA
---
**a. Basic descriptive statistics**

The categorical data in the survey is relatively limited in diversity. Each categorical 
column is as follows:
```
SATISFACTION 
Number of categories = 2 
Categories = 'satisfied', 'neutral or dissatisfied' 
'satisfied' no. = 70882 
'neutral or dissatisfied' no. = 58605 
CUSTOMER TYPE 
Number of categories = 2 
Categories = 'Loyal Customer', 'disloyal Customer' 
'Loyal Customer' no. = 105773 
'disloyal Customer' no. = 23714 
TYPE OF TRAVEL 
Number of categories = 2 
Categories = 'Business travel', 'Personal Travel' 
'Business travel' no. = 89445 
'Personal Travel' no. = 40042 
CLASS 
Number of categories = 3 
Categories = 'Business', 'Eco','Eco Plus' 
'Business' no. = 61990 
'Eco' no. = 58117 
'Eco Plus' no. = 9380
```

We have a nice mix of categorical data, though the dimensions to each column aren’t 
huge. Next, using Pandas’ .describe() function, I got an idea of the size and scope of our 
numerical columns, showing nothing significantly out of the ordinary.

![Screenshot_24-8-2024_211950_localhost](https://github.com/user-attachments/assets/a4a5a225-7bdc-443d-9755-015b60bc5e8d)


**b. Outliers**

Again, nothing seemed unnatural from our data. All ages seem perfectly normal and 
there are no unusual data points that could interfere with our goals.

**c. Categorical comparisons**

To begin EDA, I wanted to assess the relationship categorical variables had with each other. This would allow us to observe underlying trends and basic targets for the improvement of our business model, and also allow us to comment on the "importance" of a result - as a theoretical example, were 99% of loyal customers female, we would consider the predictions of a model with "female loyal customers" as its _Gender_ and _Customer Type_ attributes to be more accurate/important than a model with "male loyal customers" as its respective attributes.

A heatmap below is shown, where each square represents the percentage of _X_'s total population that rows with _Y_ as an attribute make up.

![Screenshot_24-8-2024_85721_localhost](https://github.com/user-attachments/assets/b1178218-16b7-4e7a-9b69-88a9a74a488e)

Immediately we are able to identify certain strong relationships. Some seem obvious, for instance, 96% of customers who bought Business class tickets were travelling for business, but some other interesting insights can be found. One particularly fascinating insight that practically leaps off the heatmap is the staggering percentage of disloyal customers travelling for business - 99%. Expanding upon this, 76% of disloyal customers and 32% of loyal customers were neutral or dissatisfied, which clearly suggests a noticeable relationship between customer satisfaction and loyalty, and thus one point the company could draw from this would be to target disloyal businesspeople customers in a survey and see in which specific areas the airline fails to meet their expectations. We actually have survey data to this effect in our set, and so we can explore this group's total average scores. Below are the results from a .loc method Pandas DataFrame that has used Boolean vectors to locate only disloyal businesspeople.

```
Seat comfort mean is 2.713161811609611 and kurtosis is -0.7548689958092485
Departure/Arrival time convenient mean is 2.3792898150116946 and kurtosis is -1.0904827491557103
Food and drink mean is 2.704826706357644 and kurtosis is -0.7588721010520767
Gate location mean is 2.9900914310014883 and kurtosis is -0.6863066499007835
Inflight wifi service mean is 3.044099510950457 and kurtosis is -1.2393620626584478
Cleanliness mean is 3.7001488411652135 and kurtosis is 0.007163375726185084
Online boarding mean is 3.060174356793536 and kurtosis is -1.2566803954112333
Checkin service mean is 3.2224112268764618 and kurtosis is -0.9298488465326109
Baggage handling mean is 3.6938124601318307 and kurtosis is 0.013254635013807636
Leg room service mean is 3.220837763129917 and kurtosis is -1.1533902394235567
On-board service mean is 3.2355092494152666 and kurtosis is -0.9084854449319226
Ease of Online booking mean is 3.058813523283011 and kurtosis is -1.2511706071934963
Online support mean is 3.0060812247501594 and kurtosis is -1.2928184453307967
Inflight entertainment mean is 2.714607697214544 and kurtosis is -0.7747540178244261
```
For comparison, below are the same results on the whole "airlines" data set.

```
Seat comfort average is 2.8385861128916416 and kurtosis is -0.943048565425217
Departure/Arrival time convenient average is 2.990277016225567 and kurtosis is -1.0895432090728316
Food and drink mean average is 2.852023755280453 and kurtosis is -0.986629086710654
Gate location average is 2.990377412404334 and kurtosis is -1.0896943651527216
Inflight wifi service average is 3.2491601473506995 and kurtosis is -1.121501703761247
Cleanliness average is 3.7058855329106395 and kurtosis is -0.20779510383832012
Online boarding average is 3.352545043131743 and kurtosis is -0.9378696244573876
Checkin service average is 3.3407291851691676 and kurtosis is -0.7935781983045813
Baggage handling average is 3.6954597758848378 and kurtosis is -0.23732068367928605
Leg room service average is 3.4861182975897194 and kurtosis is -0.8411766589635512
On-board service average is 3.4651432190104026 and kurtosis is -0.7846839328139485
Ease of Online booking average is 3.4721709515240913 and kurtosis is -0.9105040178682424
Online support average is 3.519967255400156 and kurtosis is -0.8096140803023193
Inflight entertainment average is 3.3837450863793275 and kurtosis is -0.5322463893664238
```
Nothing stands out from these means. Not only do they not differ greatly from each other, but the kurtoses on each set on means are all significantly leptokurtic and thus the means are not a good representation of the average score. From this, one could conclude that whilst there is nothing untoward about the service being provided to disloyal travelling businesspeople, satisfaction rates are still relatively low, and the company could take steps to improve its image amongst businesspeople and perhaps do more in-depth studies into why disloyal customers are dissatisfied.

**d. EDA conclusions**

We can see some trends within the basic categorical data itself. More people appear to 
use the company's services for business travel, though there seems to be a relatively even split of 
Business class and Economy class passengers. This alone suggests that the company could do 
more to promote the Business class amongst travelling businesspeople. 
With respect to my actual goal, though, the company does extremely well in retaining 
loyal customers, as the satisfaction level of their passengers is again evenly split, but 
customer type is heavily in favour of loyal customers. This could mean one of two 
things - either the airline brand image may simply be strong enough to keep customers 
loyal to the company without providing them with an adequate service, or 
the company is on the brink of a major exodus of previously loyal passengers.

Other than the aforementioned, for the purposes of my project, I have identified that satisfaction and loyalty does have a correlation, which is to be expected, but it is not particularly relevant in any area except for disloyalty, where the almost entirely businessperson population of disloyal customers is approximately 3/4 neutral or dissatisfied. A quick glance at the business class x row suggests that most people are satisifed with the business class service, and since almost everyone using it is a businessperson, the company could then conclude that an effective way to promote customer satisfaction in disloyal travellers and thus encourage customer retention would be to make the business class ticket more appealing to businesspeople and discourage Eco/Eco Plus class in this population.

A final, fascinating point is that these stark differences appear to form more frequently around disloyal travellers. The split in business/personal travellers is practically non-existent in disloyal travellers, whilst in loyal ones we can see a relatively balanced set. The majority satisfaction category is 14% larger in disloyal customers than in loyal ones. Only the ticket type remains somewhat even with loyal customers. With this in mind, it makes sense that any model I train would likely learn these differences and express them during the results generation stage. Thus, I will compare the output of loyal and disloyal customers in the ML results stage at the end of this project.


Methodology
---
**a. Train/validation/test split**

With my clean dataset, I first split my data set into three parts, for training, validation, 
and testing, using scikit learn’s train_test_split() function[2]. These were then stored in 
variables X_train, y_train, X_validation, y_validation, X_test, y_test respectively. Default 
settings were used other than random_state=22 which was used for consistent results. 
I also used Python’s pickle library to serialize these test objects for use in other Jupyter 
notebooks.

**b. Initial model selection**

With the samples tested for authenticity, I moved on to the initial stages of model 
selection. As the target variables are both categorical, this project is a classification 
problem. For this, three basic algorithms were selected for hyperparameter testing. 
These are Random Forest, SVM, and Gradient Boosting. I chose to implement these 
using scikit learn’s RandomForestClassifier[3], SVC[4], and XGBoost’s 
XGBoostClassifier[5]. I decided to score them based on the validation data. The results 
were as follows:
```
RF scores 
1.0 
0.986078504057004 
 
SVM scores 
0.9827695095831731 
0.977758556777462 
 
XGBoost scores 
0.9966088198143775 
0.9906915441327897
```
These aren’t indicative of the true score of the model as we haven’t tested them on the 
third test set. That will come later. For now, we observe that XGBoost performs slightly 
better.

**c. Hyperparameter testing**

Given the small scale of this project, and limited budget and time constraints, a 
relatively low number of hyperparameters were tested during the model selection 
process. These will be explained below.

**c.i. RandomForestClassifier()**

The settings tested on RandomForestClassifier were as follows:
```
max_depth: Default (unlimited) scored best. 
min_samples_split: 3 scored best, (default 2). 
min_samples_leaf: Default (1) scored best. 
weight: Default (0) scored best. 
max_leaf_nodes: Default (unlimited) scored best. 
impurity: Default (0) scored best. 
n_estimators: Default (100) did not necessarily score best, but other settings did not 
consistently beat it and so default was kept. 
Post-pruning – CCP alphas: A significant downward trend was noted on increasing of 
ccp_alpha, so I did not use it.
```
With this in mind, the RandomForestClassifier model selected was default.

**c.ii. SVC**

With other algorithms, I used all the training data to fit the model, and then all the 
validation data to validate the model. However, with SVC, the training time scales 
quadratically with the number of rows[6]. It was therefore necessary for me to sample 
the dataset. I set 10000 as an acceptable no. of rows.

The settings tested on SVC were as follows:
```
kernel: Default (rbf – radial base function) performed best. 
degree, (polynomial kernel only: All settings failed to beat rbf. 
gamma: Default (scale) performed best. 
shrinking: Default (True) performed best. 
tol: No discernable difference so tolerance was kept as default. 
cache: No discernale difference so default was kept. 
max_iter: Default (unlimited) beat all tested maximum iterations settings. 
I therefore chose SVC with default settings.
```
**c.iii. XGBoost**
The settings tested on XGBoost were as follows:
```
max_depth: Default (unlimited) performed the best. 
n_estimators: Default (100) did not consistently perform the best, but the difference 
was negligible so default was left  
learning_rate: Default (0.3) performed the best. 
max_leaves: Default (unlimited) performed the best. 
min_child_weight: Default (0.0) performed the best. 
grow_policy: Default (depthguide) was identical to lossguide, so I selected default. 
objective: Default (MSQ) was not beaten by any other objectives, so it was kept. 
eval_metric: There appears to not have been any change to the model performances 
whatsoever with this. Default is therefore kept.
```
With this in mind, XGBoost is kept as default.

**d. Cross-fold validation**

All models performed best under default settings. I then compared each model using k
fold cross validation to gain an unbiased score. The results were as follows.
```
Forest scores mean is 0.9857350771814775 
Forest score against test is 0.9863462251328309 
SVM score is 0.977758556777462 
XGBoost scores mean is 0.9901147765139224 
XGBoost score against test set is 0.990145805016681
```
XGBoost under default settings clearly beats the other classifier models, if only slightly. 
It is therefore chosen as our model.

**e. PCA and unsupervised learning**

Before assessing the results of the classifier, I decided to make use of unsupervised 
learning to see if there were any “natural” groups that formed in the set. For this, I used 
scikit learn’s PCA function[7]. I initially tested this on the whole group.
![1st PCA](https://github.com/user-attachments/assets/c72b68fa-db68-4ad3-a6ba-f021b65ba8b4)

Apart from the y axis, no clear clusters shown here. I then checked for age, and 
departure and arrival delay.
![2nd PCA](https://github.com/user-attachments/assets/2c49e41b-7f49-43b0-b12e-81feda2f3a62)


I then checked departure and arrival delay.
![3rd PCA](https://github.com/user-attachments/assets/c77cf4b4-74e6-47e7-a01e-10d3b163711c)


Finally, I assessed the “survey” columns themselves.
![4th PCA](https://github.com/user-attachments/assets/44ccd01d-54cd-425f-bff2-8fb7bc165ae2)


All PCAs seemed to suggest is that departure delay and arrival delay are weakly related, 
(unsurprisingly), but other than that, there are no clear groups formed from the data.


**f. Results generation**

The results from PCA suggest there are no notable “groups” that can be inputted into 
the classifier and I therefore had to “toggle” each feature to achieve a result. 24 
separate combinations of the categorical data were possible. These were:
```
[['satisfied', 'Female', 'Personal Travel', 'Eco'], 
['satisfied', 'Female', 'Personal Travel', 'Eco Plus'], 
['satisfied', 'Female', 'Personal Travel', 'Business'], 
['satisfied', 'Female', 'Business travel', 'Eco'], 
['satisfied', 'Female', 'Business travel', 'Eco Plus'], 
['satisfied', 'Female', 'Business travel', 'Business'], 
['satisfied', 'Male', 'Personal Travel', 'Eco'], 
['satisfied', 'Male', 'Personal Travel', 'Eco Plus'], 
['satisfied', 'Male', 'Personal Travel', 'Business'], 
['satisfied', 'Male', 'Business travel', 'Eco'], 
['satisfied', 'Male', 'Business travel', 'Eco Plus'], 
['satisfied', 'Male', 'Business travel', 'Business'], 
['neutral or dissatisfied', 'Female', 'Personal Travel', 'Eco'], 
['neutral or dissatisfied', 'Female', 'Personal Travel', 'Eco Plus'], 
['neutral or dissatisfied', 'Female', 'Personal Travel', 'Business'], 
['neutral or dissatisfied', 'Female', 'Business travel', 'Eco'], 
['neutral or dissatisfied', 'Female', 'Business travel', 'Eco Plus'], 
['neutral or dissatisfied', 'Female', 'Business travel', 'Business'], 
['neutral or dissatisfied', 'Male', 'Personal Travel', 'Eco'], 
['neutral or dissatisfied', 'Male', 'Personal Travel', 'Eco Plus'], 
['neutral or dissatisfied', 'Male', 'Personal Travel', 'Business'], 
['neutral or dissatisfied', 'Male', 'Business travel', 'Eco'], 
['neutral or dissatisfied', 'Male', 'Business travel', 'Eco Plus'], 
['neutral or dissatisfied', 'Male', 'Business travel', 'Business']]
```

My next step was to see if any numerical information could be represented more readily 
by the mean. I decided to use SciPy’s kurtosis[8] function to determine the kurtosis of 
each numerical feature “against” age. I did this by using Pandas’ .loc[] function and a 
boolean vector to “drill into” each age’s results in the original data set and collect each 
features’ mean kurtosis result.
```
Mean kurtosis is for Age to Flight Distance is 0.46463908765658923 
Mean kurtosis is for Age to Arrival Delay in Minutes is 57.497011775273485 
Mean kurtosis is for Age to Departure Delay in Minutes is 60.27792424935889 
Mean kurtosis is for Age to Seat comfort is -0.9264799631092573 
Mean kurtosis is for Age to Departure/Arrival time convenient is 
1.0682720132972 
Mean kurtosis is for Age to Food and drink is -0.981984345197727 
Mean kurtosis is for Age to Gate location is -1.0640644807015396 
Mean kurtosis is for Age to Inflight wifi service is -1.1343370043533183 
Mean kurtosis is for Age to Inflight entertainment is -0.5423705731670951 
Mean kurtosis is for Age to Online support is -0.7348773008794438 
Mean kurtosis is for Age to Ease of Online booking is -0.9243988051003394 
Mean kurtosis is for Age to On-board service is -0.8170222061786446 
Mean kurtosis is for Age to Leg room service is -0.78592773890839 
Mean kurtosis is for Age to Baggage handling is -0.2806797979963458 
Mean kurtosis is for Age to Checkin service is -0.8150683517862631 
Mean kurtosis is for Age to Cleanliness is -0.26445226681445794 
Mean kurtosis is for Age to Online boarding is -0.9509849544502398
```
Each mean kurtosis was quite leptokurtic, except for the two types of delay. For this reason, I chose those two to be represented by the mean in each "setting". I did not use the mean for the others.

With this in mind, I ran the model 24 times with each combination of categorical data, 
looping through age from 7-85 and flight distance from 1-7000 in intervals of 100. This generated an array matrix that could be interpreted as accurate predictions.

One final thing to note is whilst I “toggled” all categorical data, and looped through age 
and flight distance, I did not “toggle” survey results as the number of possible 
permutations was staggering, and, given my project constraints, I did not have the time 
nor the processing power to return the array efficiently. For this reason, I created a 
“generic score”, the results assume the customer scores the same in each survey 
column. 
For reference, I used an online permutations calculator from numbergenerator.org[9] to 
calculate that for a sample size of fourteen, 16,384 permutations were possible. This, 
compounded with our age, distance, and other categorical “toggles” would have 
resulted in a results array number in the millions, which was outside of my capstone 
scope.

**g. Feature importance**

XGBoost learned that Age and Flight Distance were the most important features in this model.

Key:
```
f0: onehotencoder__Gender_Female
f1: onehotencoder__Gender_Male
f2: onehotencoder__satisfaction_v2_neutral or dissatisfied
f3: onehotencoder__satisfaction_v2_satisfied
f4: onehotencoder__Type of Travel_Business travel
f5: onehotencoder__Type of Travel_Personal Travel
f6: onehotencoder__Class_Business
f7: onehotencoder__Class_Eco
f8: onehotencoder__Class_Eco Plus
f9: remainder__Age
f10: remainder__Flight Distance
f11: remainder__Seat comfort
f12: remainder__Departure/Arrival time convenient
f13: remainder__Food and drink
f14: remainder__Gate location
f15: remainder__Inflight wifi service
f16: remainder__Inflight entertainment
f17: remainder__Online support
f18: remainder__Ease of Online booking
f19: remainder__On-board service
f20: remainder__Leg room service
f21: remainder__Baggage handling
f22: remainder__Checkin service
f23: remainder__Cleanliness
f24: remainder__Online boarding
f25: remainder__Departure Delay in Minutes
f26: remainder__Arrival Delay in Minutes
```
![feature importance](https://github.com/user-attachments/assets/d2639467-657a-4fa4-bd19-2bec8bcbddb6)

Thus, these two were compared to observe trends.

Results and conclusion
---
**a. Results**

Not every “setting” returned disloyal customers. However, for every setting, there were 
no significant trends amongst those predicted as Loyal. I have included a small number 
of plots below, indicating Flight Distance, (in km), against age, (in years), for three 
“settings”.
![real loyal only](https://github.com/user-attachments/assets/fea7bfec-5b0c-49d8-b160-6fc55d0c99fb)


However, for those “settings” that returned disloyal customers, there were notable 
trends that emerged. 
![Mixed](https://github.com/user-attachments/assets/0d90056f-29b9-4d54-93a2-cc71af27d09f)


Whilst some “settings” don’t have any notable trends, there are clear 
repeating patterns here. There is a pattern in the Business class for both sexes with 
people between about 10 and 35, in which people are frequently disloyal to a brand if 
the Flight Distance is under 3000 miles. Then, from 35-50, this increases to around 
7000 miles. This pattern also appears to extend into the Eco Plus class, but for women 
only.


**b. Conclusion**


Whilst the depth of this relatively limited project only skims the surface of the potential 
of this dataset, it does reveal some interesting insights. 
With respect to customer loyalty, age and distance commonly form trends across 
different genders, types of ticket classes, reasons for travelling, and satisfaction status, 
but only if they are disloyal to the airline. This alone allows the airline to investigate their 
business practices further. For instance, the airline could change their advertising 
practises to target people, (men and women), travelling for business in Eco Plus or 
Business class, and consider what changes to the classes they could make to 
accommodate people in those classes more. It would also be worth considering why 
younger people are more disloyal to brands for smaller flight distances compared to 
middle aged people.


References
---
Note 1: [html: https://www.kaggle.com/datasets/johndddddd/customer-satisfaction/code]https://www.kaggle.com/datasets/johndddddd/customer-satisfaction/code

Note 2: [html: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html]https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

Note 3: [html: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html]https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

Note 4: [html: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC]https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

Note 5: [html: https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster]https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster

Note 6: [html: https://stackoverflow.com/questions/55471576/increase-speed-for-svm-with-polynomial-kernel]https://stackoverflow.com/questions/55471576/increase-speed-for-svm-with-polynomial-kernel

Note 7: [html: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html]https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

Note 8: [html: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html]https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html

Note 9: [html: https://numbergenerator.org/permutations-and-combinations/list#!numbers=14&lines=5&low=0&high=100&range=0,5&unique=false&order_matters=true&csv=csv&oddeven=&oddqty=0&sorted=false&sets=&addfilters=]https://numbergenerator.org/permutations-and-combinations/list#!numbers=14&lines=5&low=0&high=100&range=0,5&unique=false&order_matters=true&csv=csv&oddeven=&oddqty=0&sorted=false&sets=&addfilters=
