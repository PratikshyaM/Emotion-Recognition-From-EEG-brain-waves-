from classification import Classification
from preprocessing import Preprocessing

#Preprocessing.preprocess_test() #to preprocess the test eeg, extract and store features
path = 'C:/Users/Dell/Desktop/Project 4th Sem/classification/s32/'
model = 'C:/Users/Dell/Desktop/EEG_Emotion_Recognition/saved_models/svm_LALV_vs_rest_cf'
Classification.test_model(Classification,path,model)

#1= the mentioned class
#0 = otherwise