import pandas as pd
import os
import numpy as np
from os import listdir
from os.path import isfile,isdir, join
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 
import pickle

class Classification:
    def __init__(self):
        self = self
        
    def emotion_row(row):
        if(row['dominance'] < 4.5 and row['arousal'] < 4.5):
            row['emo'] = 1
        else:
            row['emo'] = 0
        return row['emo']
    
    def fetch_featurespace(self,mypath):
        X =[]
        #mypath = 'C:/Users/Dell/Downloads/Project EEG Sem3/sub jects/'
        onlydir = [d for d in listdir(mypath) if isdir(join(mypath, d))] 
        for dir_ in onlydir:
            onlyfiles = [f for f in listdir(mypath+'/'+dir_) if isfile(join(mypath+'/'+dir_+'/', f))]
            for filename in onlyfiles: 
                try:
                    trial_no = int(filename[5:7])
                except:
                    trial_no = int(filename[5:6])
                file = pd.read_excel(mypath+'/'+dir_+'/'+filename)
                file['Area_mag'] =  file['Area'].apply (lambda row: abs(complex(row.replace(" ", ""))))
        
                #print(filename, "  ", trial_no)
                file.drop(file[file['SODP_No'] > 5].index, inplace = True)
                file.drop(columns=['Area'], inplace= True)
                file.drop(columns=['Trial'], inplace= True)
                file.drop(columns=['Channel'], inplace= True)
                file.drop(columns=['SODP_No'], inplace= True)
                if 'Unnamed: 0' in file.columns:
                    file.drop(columns=['Unnamed: 0'], inplace= True)
                file.reset_index(drop = True,inplace = True)
                #print(file.shape)
                X.append(file.to_numpy()) #dataframe to ndarray #.to_numpy()

        print(np.asarray(X).shape)
        return X
    
    def fetch_target(self):
        #path_label = 'C:/Users/pratikshya.mishra/Desktop/EEG/EEG Code/subjects - Copy/Labels_Files_Binary/'
        path_label = 'C:/Users/Dell/Downloads/Project EEG Sem3/Label_Files/'
        
        files_label = os.listdir(path_label)
        
        y=[]
        for f in files_label:
            data_label = pd.read_excel(path_label + f)
            data_label['emo'] = data_label.apply(lambda row: self.emotion_row(row), axis=1)
            
            for e in data_label.emo.values:
                #y.append([e for i in range(100)])
                y.append(e)
                
        #y=np.asarray(y).reshape(1280 , 1)
        print(np.asarray(y).shape)
        return y
    
    def split_train_test(self,path):
        X =self.fetch_featurespace(self,path)
        X = np.asarray(X).reshape(880,300)
        print(np.asarray(X).shape)
        y =self.fetch_target()
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        return X_train, X_test, y_train, y_test
    
    def best_params(self,path):
        X_train, X_test, y_train, y_test = self.split_train_test(self,path)
        # defining parameter range 
        param_grid = {'C': [0.1, 1, 10, 100, 1000],  
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                      'kernel': ['rbf']}  
          
        grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3) 
          
        # fitting the model for grid search 
        grid.fit(X_train, y_train) 
        # print best parameter after tuning 
        print(grid.best_params_) 
          
        # print how our model looks after hyper-parameter tuning 
        print(grid.best_estimator_) 
        return grid.best_estimator_, X_train, X_test, y_train, y_test
    
    def train_model(self,path):
        clf,X_train, X_test, y_train, y_test = self.best_params(self,path)
        clf.fit(X_train, y_train)
        
        #Predict the response for test dataset
        y_pred = clf.predict(X_train)
        
        #Import scikit-learn metrics module for accuracy calculation
        from sklearn import metrics
        print("Accuracy:",metrics.accuracy_score(y_train, y_pred))
        print("F1_Score:",metrics.f1_score(y_train, y_pred, labels=np.unique(y_pred),  average = 'micro'))
        print("F1_Score:",metrics.f1_score(y_train, y_pred, labels=np.unique(y_pred),  average = 'macro'))
        print("F1_Score:",metrics.f1_score(y_train, y_pred, labels=np.unique(y_pred),  average = 'weighted'))
        
        y_pred = clf.predict(X_test)
        #Import scikit-learn metrics module for accuracy calculation
        from sklearn import metrics
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        print("F1_Score:",metrics.f1_score(y_test, y_pred, labels=np.unique(y_pred),  average = 'micro'))
        print("F1_Score:",metrics.f1_score(y_test, y_pred, labels=np.unique(y_pred),  average = 'macro'))
        print("F1_Score:",metrics.f1_score(y_test, y_pred, labels=np.unique(y_pred),  average = 'weighted'))
        
        with open ('svm_HAHV_vs_rest_cf','wb') as f:  # to store or save model - write mode
            pickle.dump(clf,f)
            
    def test_model(self,path,model):
        #model = 'svm_VD_cf'
        #path = 'C:/Users/Dell/Desktop/Project 4th Sem/classification/s32/'
        X =self.fetch_featurespace(self,path)
        X = np.asarray(X).reshape(1,300)
        with open(model,'rb') as f: # to load the saved model - read mode
            mp = pickle.load(f)

        y_pred = mp.predict(X)
        print(y_pred)