from sklearn.datasets import load_files
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import matplotlib.patches as mpatches
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import model_selection
import warnings
warnings.filterwarnings("ignore")

def main():
    d = pd.read_csv('Hatecomments.csv')
    x= d['Comments']; y=d ['Target']   
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
    Xtrain,Xtest = preproc(xtrain,xtest)
    classify(Xtrain,ytrain,Xtest,ytest)
    
   

def preproc(xtrain,xtest):
    xtrain_1=[]
    xtest_1 =[]   
    stemmer = PorterStemmer()
    for X in xtrain:
        Y=str(X).replace('\n','')       
        X=WhitespaceTokenizer().tokenize(str(Y))
        X=re.sub(r'[^\w]', ' ', str(X))
        X=word_tokenize(str(X))
        stems = [stemmer.stem(token) for token in X]
        xtrain_1.append(str(stems))   
    for X in xtest:
        Y=str(X).replace('\n','')    
        X=WhitespaceTokenizer().tokenize(str(Y))
        X=re.sub(r'[^\w]', ' ', str(X))
        X=word_tokenize(str(X))
        stems = [stemmer.stem(token) for token in X]
        xtest_1.append(str(stems))  
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.2)
    Xtrain = vectorizer.fit_transform(xtrain_1)
    Xtest = vectorizer.transform(xtest_1)
    return (Xtrain,Xtest)


def cross_validate_LR(Xtrain,ytrain,Xtest,ytest):
    penalty = ['l2']
    C = [0.01,0.1,1,10,100]
    accuracy_value = []
    for Ci in C:
        model = LogisticRegression(verbose=0, solver='saga', C=Ci,penalty='l2', max_iter=100000)
        model.fit(Xtrain, ytrain)
        preds = model.predict(Xtest)
        accuracy_value.append(metrics.accuracy_score(ytest, preds))
    plt.errorbar(C,accuracy_value)
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title('Linear Regression C vs Accuracy')
    plt.show()

def cross_validate_dec_tree(Xtrain,ytrain,Xtest,ytest):
    accuracy_value = []
    depth_range = [1,5,10,15,20,25]
    for depth_i in depth_range:
        model =DecisionTreeClassifier(max_depth=depth_i)
        model.fit(Xtrain, ytrain)
        preds = model.predict(Xtest)
        accuracy_value.append(metrics.accuracy_score(ytest, preds)) 
    plt.errorbar(depth_range,accuracy_value)
    plt.xlabel('Max_Depth_i')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Max_Depth_i vs Accuracy')
    plt.show()

def cross_validate_knn(Xtrain,ytrain,Xtest,ytest):
    accuracy_value = []
    ki_range = [1,2,3,4,5,6]
    for ki in ki_range:
        model = KNeighborsClassifier(n_neighbors=ki)
        model.fit(Xtrain, ytrain)
        preds = model.predict(Xtest)
        accuracy_value.append(metrics.accuracy_score(ytest, preds))
    plt.errorbar(ki_range,accuracy_value)
    plt.xlabel('ki')
    plt.ylabel('Accuracy')
    plt.title('KNN ki vs Accuracy')
    plt.show()

def cross_validate_randforest(Xtrain,ytrain,Xtest,ytest):
    accuracy_value = []
    depth_range = [1,5,10,15,20,25]
    for depth_i in depth_range:
        model = RandomForestClassifier(max_depth=depth_i, random_state=0)
        model.fit(Xtrain, ytrain)
        preds = model.predict(Xtest)
        accuracy_value.append(metrics.accuracy_score(ytest, preds))  
    plt.errorbar(depth_range,accuracy_value)
    plt.xlabel('Max_Depth_i')
    plt.ylabel('Accuracy')
    plt.title('Random forest Max_Depth_i vs Accuracy')
    plt.show()

def cross_validate_LinearSVC(Xtrain,ytrain,Xtest,ytest):
    accuracy_value = []
    Ci_range = [0.01,0.1,1,10,100]
    for Ci in Ci_range:
        model = LinearSVC(C=Ci)
        model.fit(Xtrain, ytrain)
        preds = model.predict(Xtest)
        accuracy_value.append(metrics.accuracy_score(ytest, preds))   
    plt.errorbar(Ci_range,accuracy_value)
    plt.xlabel('Ci')
    plt.ylabel('Accuracy')
    plt.title('LinearSVC Ci vs Accuracy')
    plt.show()
    
def classify(Xtrain,ytrain,Xtest,ytest):

    cross_validate_LR(Xtrain,ytrain,Xtest,ytest)
    model1 =  LogisticRegression()
    model1.fit(Xtrain, ytrain)
    preds1 = model1.predict(Xtest)
    results_kfold = model_selection.cross_val_score(model1, Xtrain,ytrain, cv=10)
    print("Accuracy using %s : %.2f%%" % ("Logistic Regression",results_kfold.mean()*100.0))
    lr_fpr, lr_tpr, t= roc_curve(ytest,model1.predict_proba(Xtest)[:,1])
    lr_auc = roc_auc_score(ytest, model1.predict_proba(Xtest)[:,1])
    
    

    model2 = BernoulliNB()
    model2.fit(Xtrain, ytrain)
    preds2 = model2.predict(Xtest)
    results_kfold = model_selection.cross_val_score(model2, Xtrain,ytrain, cv=10)
    print("Accuracy using %s : %.2f%%" % ("Naive Bayes",results_kfold.mean()*100.0))
    nb_fpr, nb_tpr, t = roc_curve(ytest,model2.predict_proba(Xtest)[:,1])
    nb_auc = roc_auc_score(ytest, model2.predict_proba(Xtest)[:,1])
    
    

    cross_validate_dec_tree(Xtrain,ytrain,Xtest,ytest)
    model3 =DecisionTreeClassifier(max_depth=5)
    model3.fit(Xtrain, ytrain)
    preds3 = model3.predict(Xtest)
    results_kfold = model_selection.cross_val_score(model3, Xtrain,ytrain, cv=10)
    print("Accuracy using %s : %.2f%%" % ("Decision Tree",results_kfold.mean()*100.0))
    dt_fpr, dt_tpr, _ = roc_curve(ytest,model3.predict_proba(Xtest)[:,1])
    dt_auc = roc_auc_score(ytest, model3.predict_proba(Xtest)[:,1])
    
    
    
    cross_validate_knn(Xtrain,ytrain,Xtest,ytest)
    model4 = KNeighborsClassifier(n_neighbors=2)
    model4.fit(Xtrain, ytrain)
    preds4 = model4.predict(Xtest)
    results_kfold = model_selection.cross_val_score(model4, Xtrain,ytrain, cv=10)
    print("Accuracy using %s : %.2f%%" % ("KNN",results_kfold.mean()*100.0))
    knn_fpr, knn_tpr, t = roc_curve(ytest,model4.predict_proba(Xtest)[:,1])
    knn_auc = roc_auc_score(ytest, model4.predict_proba(Xtest)[:,1])
    
    
    
    cross_validate_LinearSVC(Xtrain,ytrain,Xtest,ytest)
    model5 = LinearSVC(C=1000)
    model5.fit(Xtrain, ytrain)
    preds5 = model5.predict(Xtest)
    results_kfold = model_selection.cross_val_score(model5, Xtrain,ytrain, cv=10)
    print("Accuracy using %s : %.2f%%" % ("Linear SVC",results_kfold.mean()*100.0))
    svc_fpr, svc_tpr,t = roc_curve(ytest,model5._predict_proba_lr(Xtest)[:,1])
    svc_auc = roc_auc_score(ytest, model5._predict_proba_lr(Xtest)[:,1])
    
    
    cross_validate_randforest(Xtrain,ytrain,Xtest,ytest)
    model6 = RandomForestClassifier(max_depth=2, random_state=0)
    model6.fit(Xtrain, ytrain)
    preds6 = model6.predict(Xtest)
    results_kfold = model_selection.cross_val_score(model6, Xtrain,ytrain, cv=10)
    print("Accuracy using %s : %.2f%%" % ("Random Forest",results_kfold.mean()*100.0))
    rf_fpr, rf_tpr, t = roc_curve(ytest,model6.predict_proba(Xtest)[:,1])
    rf_auc = roc_auc_score(ytest, model6.predict_proba(Xtest)[:,1])
    

    plt.plot(lr_fpr,lr_tpr,label= "{} , AUC={:.3f}".format("Logistic classifier",lr_auc),color='red')
    plt.plot(nb_fpr,nb_tpr,label= "Naive Bayes",color='blue')
    plt.plot(dt_fpr, dt_tpr,label= "Decision tree",color='black')
    plt.plot(knn_fpr, knn_tpr,label= "Kneighbors",color='yellow')
    plt.plot(svc_fpr, svc_tpr,label= "SVC",color='cyan')
    plt.plot(rf_fpr, rf_tpr,label= "Random Forest",color='pink')
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    red_patch = mpatches.Patch(color='red', label="{} , AUC={:.3f}".format("Logistic classifier",lr_auc))
    blue_patch = mpatches.Patch(color='blue', label="{} , AUC={:.3f}".format("Naive Bayes",nb_auc))
    black_patch = mpatches.Patch(color='black', label="{} , AUC={:.3f}".format("Decision Tree",dt_auc))
    yellow_patch = mpatches.Patch(color='yellow', label="{} , AUC={:.3f}".format("KNN",knn_auc))
    cyan_patch = mpatches.Patch(color='cyan', label="{} , AUC={:.3f}".format("SVC",svc_auc))
    pink_patch = mpatches.Patch(color='pink', label="{} , AUC={:.3f}".format("Random Forest",rf_auc))
    plt.legend(handles=[red_patch, blue_patch,black_patch,yellow_patch,cyan_patch,pink_patch])
    plt.plot([0, 1], [0, 1], color="green",linestyle="--")
    plt.title("ROC curves")
    plt.show()
    



if __name__=="__main__":
    main()


