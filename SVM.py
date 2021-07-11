#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 00:06:19 2020

@author: nooreen
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from matplotlib import style
style.use("ggplot")


fread = open("/users/nooreen/Downloads/spambase.data", mode="r")
data = np.loadtxt(fread, delimiter=",")

ndoc = data.shape[0]
idf = np.log10(ndoc/(data != 0).sum(0))
X= data/100.0*idf

np.random.shuffle(data)
Y = data[:,57] 
X = data[:,:54] 

#kERNALS
classifier = SVC(kernel="linear")
scores_ln = cross_val_score(classifier, X, Y, cv = 10, n_jobs= 8)
mean_l=scores_ln.mean()
var_l=scores_ln.var()
sd_l=scores_ln.std()
print("Mean Linear Kernel: {}".format(mean_l))
print("Variance Linear Kernel: {}".format(var_l))
print("Standard deviation Linear Kernel: {}".format(sd_l))


clf_pl = SVC(kernel="poly", degree = 2)
scores_pl = cross_val_score(clf_pl, X, Y, cv = 10, n_jobs= 8)
mean_pl=scores_pl.mean()
var_pl=scores_pl.var()
sd_pl=scores_pl.std()
print("Mean polynomial Kernel: {}".format(mean_pl))
print("Variance polynomial Kernel: {}".format(var_pl))
print("Standard deviatiion polynomial Kernel: {}".format(sd_pl))

clf_rbf = SVC(kernel="rbf")
scores_rbf = cross_val_score(clf_rbf, X, Y, cv = 10, n_jobs= 8)
mean_rbf=scores_rbf.mean()
var_rbf=scores_rbf.var()
sd_rbf=scores_rbf.std()
print("Mean RBF Kernel: {}".format(mean_rbf))
print("Variance RBF Kernel: {}".format(var_rbf))
print("Standard deviatiion RBF Kernel: {}".format(sd_rbf))


norms = np.sqrt(((X+1e-128) ** 2).sum(axis=1, keepdims=True))
XX = np.where(norms > 0.0, X / norms, 0.)

#angular kernals
clf_a = SVC(kernel="linear")
scores_ln_a = cross_val_score(clf_a, XX, Y, cv = 10, n_jobs= 8)
mean_ln_a=scores_ln_a.mean()
var_ln_a=scores_ln_a.var()
sd_ln_a=scores_ln_a.std()
print("Mean Anglar Linear Kernel: {}".format(mean_ln_a))
print("Variance Angular Linear Kernel: {}".format(var_ln_a))
print("Standard deviatiion  Angular Linear  Kernel: {}".format(sd_ln_a))


clf_poly_a = SVC(kernel="poly", degree = 2)
scores_pl_a = cross_val_score(clf_poly_a, XX, Y,cv = 10, n_jobs= 8)
mean_pl_a=scores_pl_a.mean()
var_pl_a=scores_pl_a.var()
sd_pl_a=scores_pl_a.std()
print("Mean Anglar polynomial Kernel: {}".format(mean_pl_a))
print("Variance Angular polynomial Kernel: {}".format(var_pl_a))
print("Standard deviatiion  Angular polynomial  Kernel: {}".format(sd_pl_a))


clf_rbf_a = SVC(kernel="rbf")
scores_rbf_a = cross_val_score(clf_rbf_a, XX, Y, cv = 10, n_jobs= 8)
mean_rbf_a=scores_rbf_a.mean()
var_rbf_a=scores_rbf_a.var()
sd_rbf_a=scores_rbf_a.std()
print("Mean Anglar RBF Kernel: {}".format(mean_rbf_a))
print("Variance Angular RBF Kernel: {}".format(var_rbf_a))
print("Standard deviatiion  Angular RBF  Kernel: {}".format(sd_rbf_a))


#normalization
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
xx_train, xx_test, yy_train, yy_test = train_test_split(XX, Y, test_size=0.3)
n = XX.shape[0]
n_train = xx_train.shape[0]

clf_lna = clf_a.fit(xx_train, yy_train)
clf_ln = classifier.fit(x_train, y_train)
print(str(clf_lna.support_vectors_[0]))
print(str(clf_lna.support_vectors_[0]))
def model(classifier):
    clf_fit = classifier.fit(xx_train, yy_train)
    print(str(clf_fit.score(xx_train, yy_train)))
    print(str(clf_lna.n_support_)) 
    print(str(clf_lna.support_vectors_)) 

model(clf_pl)
model(clf_poly_a)
model(clf_rbf)
model(clf_rbf_a)
