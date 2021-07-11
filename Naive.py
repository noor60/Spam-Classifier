#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:31:24 2020

@author: nooreen
"""
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split



class Classifier(BaseEstimator):

    def score(self, X, Y):
        p_x_spam_i = (2*np.pi*self.var_spam)**(-1./2) * np.exp(-1./(2*self.var_spam)*np.power(X-self.mu_spam,2))
        p_x_ham_i = (2*np.pi*self.var_ham)**(-1./2) * np.exp(-1./(2*self.var_ham)*np.power(X-self.mu_ham,2))
        
        p_x_spam = np.prod(p_x_spam_i, axis= 1)
        p_x_ham = np.prod(p_x_ham_i, axis= 1)
        
        p_spam_x = p_x_spam * self.p_spam
        p_ham_x = p_x_ham * self.p_ham
                           
        predicted_labels = np.argmax([p_ham_x,p_spam_x], axis = 0)
        return np.mean(predicted_labels == Y)

    def fit(self, X, Y, **kwargs):
        self.spam = X[Y == 1,:54]
        self.ham = X[Y == 0,:54]
        
        self.N = float(self.spam.shape[0] + self.ham.shape[0])
        self.k_spam = self.spam.shape[0] 
        self.k_ham = self.ham.shape[0] 

        self.p_spam = self.k_spam/self.N
        self.p_ham = self.k_ham/self.N
        
        self.mu_spam = np.mean(self.spam, axis=0)
        self.mu_ham = np.mean(self.ham, axis=0)
        
        self.var_spam = np.var(self.spam, axis=0)+1e-128
        self.var_ham = np.var(self.ham, axis=0)+1e-128
def evaluation_model(data, classifier, run = 10):
    scores = np.array([])
    for i in range(run):
        np.random.shuffle(data)
        Y = email[:,57] 
        X = email[:,:54]
        scores = np.append(scores,cross_val_score(classifier, X, Y, cv = 10))
    return scores
fread = open("/users/nooreen/Downloads/spambase.data", "r")
email = np.loadtxt(fread, delimiter=",")
np.random.shuffle(email)# shuffle dataset
Y = email[:,57] 
X = email[:,:54]

myclassifier= Classifier()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
scores = cross_val_score(myclassifier, X, Y, cv = 10)


scores = cross_val_score(myclassifier, X, Y, cv = 10)
print("Mean Accuracy: " + str(scores.mean())+"\n")
print("Variance Accuracy: " + str(scores.var()) +" / " +str(scores.std())+"\n")
scores_run = evaluation_model(email, myclassifier,run = 20)
print("Mean Accuracy: " + str(scores_run.mean())+"\n")
print("Variance Accuracy: " + str(scores_run.var()) +" / " +str(scores_run.std())+"\n")

md = myclassifier.fit(x_train,y_train)
print("Accuracy: "+str(myclassifier.score(x_test, y_test)))
print("p(spam): "+str(myclassifier.p_spam))
print("p(ham): "+str(myclassifier.p_ham))
