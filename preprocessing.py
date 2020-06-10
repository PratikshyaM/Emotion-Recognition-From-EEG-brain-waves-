import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy import interpolate
import numpy as np
import math
import cmath
import pandas as pd
import pickle
import time 
from os import listdir
from os.path import isfile, join

class Preprocessing:
    def __init__(self):
        self = self
    def create_directory(name):
        try:
            #Create target Directory
            os.mkdir(name)
            print("Directory ",name,"Created")
        except FileExistsError:
            print("Directory",name,"already exists")
        return name
    
    def save_fig(x,y,color,title,dir_):
        plt.figure() #figsize=(10,20)
        plt.plot(x,y,color)
        plt.title(title)
        plt.savefig(dir_)
        plt.close()
    
    def imf_create(t,data):
        mins = signal.argrelmin(data)[0]
        mins_ = [float(data*60/8064) for data in mins]
        maxs = signal.argrelmax(data)[0]
        maxs_ = [float(data*60/8064) for data in maxs]
        #extrema = np.concatenate((mins, maxs))
        spl_min = interpolate.CubicSpline(mins_, data[mins])#, bc_type = 'natural') #clamped
        #l_env = spl_min(t)
        
        spl_max = interpolate.CubicSpline(maxs_, data[maxs])#, bc_type = 'natural')#clamped
        #u_env = spl_max(t)
        mid = (spl_max(t)+spl_min(t))/2
        
        #plt.figure()
        #plt.plot(t,data)
        #plt.plot(t,l_env,'-')
        #plt.plot(t,u_env,'-')
        #plt.plot(t, mid, '--')
        #plt.title('Plottings')
        #plt.show()
        
        return data-mid
    
    def stopping_conditions(imf,t):
        mins = signal.argrelmin(imf)[0]
        mins_ = [float(data*60/8064) for data in mins]
        maxs = signal.argrelmax(imf)[0]
        maxs_ = [float(data*60/8064) for data in maxs]
        
        spl_min = interpolate.CubicSpline(mins_,imf[mins])#, bc_type = 'natural') #clamped
        spl_max = interpolate.CubicSpline(maxs_, imf[maxs])#, bc_type = 'natural')#clamped
        
        mean_amplitude = [np.abs(spl_max(i)+spl_min(i))/2 for i in range(0,len(t))]
        envelope_amplitude = [np.abs(spl_max(i)- spl_min(i))/2 for i in range(0,len(t))]
        bo = [(m/e > 0.05) for m,e in zip(mean_amplitude,envelope_amplitude)]
        
        #at each point, mean_amplitude < THRESHOLD2*envelope_amplitude
        condition = [not(m < 0.5*e) for m,e in zip(mean_amplitude,envelope_amplitude)]
        
        #mean of boolean array {(mean_amplitude)/(envelope_amplitude) > THRESHOLD} < TOLERANCE
        if((1 in condition) or (not(np.mean(bo)<0.05))):
            return False
        
        # |#zeros-#extrema|<=1
        zero_crossings = np.where(np.diff(np.signbit(imf)))[0]
        diff_zeroCr_extremas = np.abs(len(maxs)+len(mins)-len(zero_crossings))
        if(diff_zeroCr_extremas <= 1):# and mean <0.1):
            return True
        else:
            return False
               
    def calc_area_of_sodp(X,Y,i,channel):
        #Area of Second Order Difference Plot
        SX = math.sqrt(np.sum(np.multiply(X,X))/len(X))
        SY = math.sqrt(np.sum(np.multiply(Y,Y))/len(Y))
        SXY = np.sum(np.multiply(X,Y))/len(X)
        D = cmath.sqrt((SX*SX) + (SY*SY) - (4*(SX*SX*SY*SY - SXY*SXY)))
        a = 1.7321 *cmath.sqrt(SX*SX + SY*SY + D)
        b = 1.7321 * cmath.sqrt(SX*SX + SY*SY - D)
        Area = math.pi *a *b
        print("Channel=  ",channel,"Area of SODP of IMF number= ",i, " is ", Area)
        return Area
    
    def calc_mean_and_ctm(X,Y,i,channel):
        features = pd.DataFrame(columns=['radius','mean_distance','central_tendency_measure'])
        r = 0.5
        d = [ math.sqrt(X[i]*X[i] + Y[i]*Y[i]) for i in range (0,len(X))]
        delta = [1 if i<r else 0 for i in d]
        d = [i for i in d if i<r]
            
        ctm = np.sum(delta[:-2])/(len(delta)-2)
        mean_distance = np.mean(d)
        
        features.loc[0] = [r] + [ctm] + [mean_distance]
        return features
    
    def second_order_difference_plot(self,y, i, channel,dir_,imp_features,trial):
        #remove outliers
        upper_quartile = np.percentile(y,80)
        lower_quartile = np.percentile(y,20)
        IQR = (upper_quartile - lower_quartile) * 1.5
        quartileSet = (lower_quartile- IQR, upper_quartile +IQR)
        y = y[np.where((y >= quartileSet[0]) & (y <= quartileSet[1]))]
        
        #plotting SODP
        X = np.subtract(y[1:],y[0:-1]) #x(n+1)-x(n)
        Y = np.subtract(y[2:],y[0:-2]).tolist()#x(n+2)-x(n-1)
        Y.extend([0])
        self.save_fig(X,Y,'.','SODP'+str(i),dir_+'/SODP'+str(i)+'.png')
        
        Area = self.calc_area_of_sodp(X,Y,i,channel)
        features =self.calc_mean_and_ctm(X,Y,i,channel)
        
        df = pd.DataFrame({"Trial":trial,"Channel":channel,"SODP_No":i,"Area":Area,
                           "m(r=0.5)":features[features['radius']==0.5]['mean_distance'],
                           "ctm(r=0.5)":features[features['radius']==0.5]['central_tendency_measure']})
        imp_features = imp_features.append(df,ignore_index=True)
        return imp_features
    
    def emd_algorithm(self,channel,s,t,dir_,imp_features,trial):
        #save_fig(t,s,'g',"Original Signal",dir_+'/Original Signal.png')
        r = []
        #imf = []
        r.append(s)
        i = 1
        while(True):
            if(i==7):
                break;
            h = []
            j = 1
            h.append(r[i-j])
            while(True):
                h.append(self.imf_create(t,h[j-1]))
                if(self.stopping_conditions(h[j],t)):
                    #imf.append(h[j])
                    #save_fig(t,h[j],'g','IMF'+str(i),dir_+'/IMF'+str(i)+'.png')
                    imp_features = self.second_order_difference_plot(self,h[j],i,channel,dir_,imp_features,trial)
                    break;
                else:
                    j = j+1
            r.append( r[i-1] - h[j])
            if(len(signal.argrelmin(r[i])[0])<2 or len(signal.argrelmax(r[i])[0])<2):
                #save_fig(t,r[i],'r','Residue',dir_+'/Residue.png')
                break;
            else:
                i = i+1
        
        return imp_features
    
    def preprocess_train(self,path):
        #path = 'C:/Users/Dell/Desktop/EEG_Emotion_Recognition/data/'
        fname = [d for d in listdir(path) if isfile(join(path, d))] 
        print(fname)
        for fn in fname:
            x = pickle.load(open(path+fn,'rb'),encoding = 'iso-8859-1')
            data = x['data']
            #print(data.shape)
            #labels = x['labels']
            imp_features = pd.DataFrame(columns=['Trial','Channel','SODP_No','Area','m(r=0.5)','ctm(r=0.5)'])
            
            channels =[1,2,3,4,7,11,12,14,15,16,17,18,19,20,21,24,25,29,30,32]   
            #channels =[1]   
            t = np.linspace(0,60,8064) #Since mentioned the signal is for 60 seconds
            
            curr_dir = self.create_directory(fn[-7:-4])
            print("Currently in ", curr_dir," folder.")
            
            start = time.clock()  
            
            for trial in range (1,41):#(1,41)
                dir_  = self.create_directory(curr_dir +'/Trial '+ str(trial))
                print("Currently in ", dir_," directory.")
                    
                for channel in channels:
                    dirc_  = self.create_directory(dir_ +'/Channel '+ str(channel))
                    print("Currently in ", dirc_," directory.")
                    s = data[trial-1,channel-1,:]
                    imp_features = self.emd_algorithm(self,channel,s,t, dirc_,imp_features,trial)
            
                writer = pd.ExcelWriter(dir_+'/Trial'+str(trial)+'.xlsx')
                imp_features.to_excel(writer, index=False)
                writer.save()
                imp_features.drop(imp_features.index, inplace=True)#empty dataframe for next iteration
            
            elapsed = time.clock()   
            print ("Time spent in function is: ", (elapsed-start)/60 , " mins")
            
    def preprocess_test(self,path):
        #path = 'C:/Users/Dell/Desktop/EEG_Emotion_Recognition/data_test/'
        fname = [d for d in listdir(path) if isfile(join(path, d))] 
        print(fname)
        for fn in fname:
            x = pickle.load(open(path+fn,'rb'),encoding = 'iso-8859-1')
            data = x['data']
            #print(data.shape)
            #labels = x['labels']
            imp_features = pd.DataFrame(columns=['Trial','Channel','SODP_No','Area','m(r=0.5)','ctm(r=0.5)'])
            #channels =[1,2,3,4,7,11,12,14,15,16,17,18,19,20,21,24,25,29,30,32]   
            channels =[1]   
            t = np.linspace(0,60,8064) #Since mentioned the signal is for 60 seconds
            curr_dir = self.create_directory(fn[-7:-4])
            print("Currently in ", curr_dir," folder.")
            start = time.clock()  
            for trial in range (1,2):#(1,41)
                dir_  = self.create_directory(curr_dir +'/Trial '+ str(trial))
                print("Currently in ", dir_," directory.")
                for channel in channels:
                    dirc_  = self.create_directory(dir_ +'/Channel '+ str(channel))
                    print("Currently in ", dirc_," directory.")
                    s = data[trial-1,channel-1,:]
                    imp_features = self.emd_algorithm(self,channel,s,t, dirc_,imp_features,trial)
            
                writer = pd.ExcelWriter(dir_+'/Trial'+str(trial)+'.xlsx')
                imp_features.to_excel(writer, index=False)
                writer.save()
                imp_features.drop(imp_features.index, inplace=True)#empty dataframe for next iteration
            elapsed = time.clock()   
            print ("Time spent in function is: ", (elapsed-start)/60 , " mins")
