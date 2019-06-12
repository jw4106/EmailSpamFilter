Email Spam Classification using Naive Bayes & Support Vector Machine
--------------------------------------------------------------------

### Introduction

This project is meant to imitate or even improve upon current spam email classifiers based off of old Natural Language  Processing homeworks for queies and documents. The use of feature vectors along with Sklearn libraries allowed for us to compare the Naive Bayes multinomial algorithm with the Support Vector Machine algorithm to determine the main differences and subtleties between the two in spam detection. We used a compilation from Ling-spam (http://www.aueb.gr/users/ion/data/lingspam_public.tar.gz ) corpus to generate a training folder with half spam and half ham(Not spam) to train our system. Once trained, we ran a test batch to see how our program ran.

### Executions

*To run our program type:* python3 spamfilter.py

Output will be in the form of: 

![alt text][https://github.com/jw4106/EmailSpamFilter/blob/master/nlp_graphs.png]

### Techniques 

We implemented feature vectors to calculate the features of a 2d matrix, using rows as emails and columns as common words, moreover we used the stop words from the query and documents homework so that we do not use common words in our dictionary to give us false
keywords. Algorithms we currently have implemented are the Naive Bayes Multinomial and Support Vector Machine Algorithm.

### Future Plans 

We want to move over from technical to analytical to further understand what people classify as spam such as emails that are scams and emails that are undesirable based on the user. We want to be able to implement a spam filter which understands the type of “person” that is using it as to further more accurately filter desirable and undesirable emails. Furthermore, we want to the errors between the two algorithms we have so far such as the risk of flagging ham emails from spam and how to mitigate these errors by using the analysis we mentioned above.
