#--------------------------------
# UNIVARIABLE REGRESSION EXAMPLE
#    -USING SciPy FOR OPTIMIZATION
#--------------------------------

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from   scipy.optimize import minimize
import copy
import os


#------------------------
#CODE PARAMETERS
#------------------------

#USER PARAMETERS
IPLOT=True
os.chdir('590-sl1777/HW2.1')  # change the working directory into the howmework folder
INPUT_FILE='weight.json'
FILE_TYPE="json"


OPT_ALGO='BFGS'	#HYPER-PARAM

#UNCOMMENT FOR VARIOUS MODEL CHOICES (ONE AT A TIME)
model_type="logistic"; NFIT=4; X_KEYS=['x']; Y_KEYS=['y']


#RANDOM INITIAL GUESS FOR FITTING PARAMETERS
p=np.random.uniform(0.5,1.,size=NFIT)


#------------------------
#DATA CLASS
#------------------------

class DataClass:

    #INITIALIZE
    def __init__(self,FILE_NAME):
        global p
        global method
        if(FILE_TYPE=="json"):

            #READ FILE
            with open(FILE_NAME) as f:
                self.input = json.load(f)  #read into dictionary

            #CONVERT DICTIONARY INPUT AND OUTPUT MATRICES #SIMILAR TO PANDAS DF   
            X=[]; Y=[]
            for key in self.input.keys():
                if(key in X_KEYS): X.append(self.input[key])
                if(key in Y_KEYS): Y.append(self.input[key])

            #MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
            self.X=np.transpose(np.array(X))
            self.Y=np.transpose(np.array(Y))
            self.been_partitioned=False

            #INITIALIZE FOR LATER
            self.YPRED_T=1; self.YPRED_V=1

            #TAKE MEAN AND STD DOWN COLUMNS (I.E DOWN SAMPLE DIMENSION)
            self.XMEAN=np.mean(self.X,axis=0); self.XSTD=np.std(self.X,axis=0) 
            self.YMEAN=np.mean(self.Y,axis=0); self.YSTD=np.std(self.Y,axis=0) 
        else:
            raise ValueError("REQUESTED FILE-FORMAT NOT CODED"); 

    def report(self):
        print("--------DATA REPORT--------")
        print("X shape:",self.X.shape)
        print("X means:",self.XMEAN)
        print("X stds:" ,self.XSTD)
        print("Y shape:",self.Y.shape)
        print("Y means:",self.YMEAN)
        print("Y stds:" ,self.YSTD)

    def partition(self,f_train=0.8, f_val=0.15,f_test=0.05):
        #TRAINING: 	 DATA THE OPTIMIZER "SEES"
        #VALIDATION: NOT TRAINED ON BUT MONITORED DURING TRAINING
        #TEST:		 NOT MONITORED DURING TRAINING (ONLY USED AT VERY END)


        if(f_train+f_val+f_test != 1.0):
            raise ValueError("f_train+f_val+f_test MUST EQUAL 1");

        #PARTITION DATA
        rand_indices = np.random.permutation(self.X.shape[0])
        CUT1=int(f_train*self.X.shape[0]); 
        CUT2=int((f_train+f_val)*self.X.shape[0]); 
        self.train_idx, self.val_idx, self.test_idx = rand_indices[:CUT1], rand_indices[CUT1:CUT2], rand_indices[CUT2:]
        self.been_partitioned=True

    def model(self,x,p):
        return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.0001))))

    def predict(self):
        self.YPRED_T=self.model(self.X[self.train_idx],self.p)
        self.YPRED_V=self.model(self.X[self.val_idx],self.p)
        self.YPRED_TEST=self.model(self.X[self.test_idx],self.p)

    def normalize(self):
        self.X=(self.X-self.XMEAN)/self.XSTD 
        self.Y=(self.Y-self.YMEAN)/self.YSTD  

    def un_normalize(self):
        self.X=self.XSTD*self.X+self.XMEAN 
        self.Y=self.YSTD*self.Y+self.YMEAN 
        self.YPRED_V=self.YSTD*self.YPRED_V+self.YMEAN 
        self.YPRED_T=self.YSTD*self.YPRED_T+self.YMEAN 
        self.YPRED_TEST=self.YSTD*self.YPRED_TEST+self.YMEAN 

    #------------------------
    #DEFINE LOSS FUNCTION
    #------------------------

    def loss(self,p,data):   
        y_pred = self.model(self.X[data],p)
        y_actual = self.Y[data]
        residual = y_pred - y_actual
        return np.mean(np.power(residual,2))
      


            

    def fit(self):
        residual = self.optimize('GD',0.001,method = method)
        popt = residual
        print("OPTIMAL PRAM:",popt)
        self.p = popt

    #FUNCTION PLOTS
    def plot_1(self,xla='x',yla='y'):
        if(IPLOT):
            fig, ax = plt.subplots()
            ax.plot(self.X[self.train_idx]    , self.Y[self.train_idx],'o', label='Training') 
            ax.plot(self.X[self.val_idx]      , self.Y[self.val_idx],'x', label='Validation') 
            ax.plot(self.X[self.test_idx]     , self.Y[self.test_idx],'*', label='Test') 
            ax.plot(self.X[self.train_idx]    , self.YPRED_T,'.', label='Model') 
            plt.xlabel(xla, fontsize=18);	plt.ylabel(yla, fontsize=18); 	plt.legend()
            plt.show()

    #PARITY PLOT
    def plot_2(self,xla='y_data',yla='y_predict'):
        if(IPLOT):
            fig, ax = plt.subplots()
            ax.plot(self.Y[self.train_idx]  , self.YPRED_T,'*', label='Training') 
            ax.plot(self.Y[self.val_idx]    , self.YPRED_V,'*', label='Validation') 
            ax.plot(self.Y[self.test_idx]    , self.YPRED_TEST,'*', label='Test') 
            plt.xlabel(xla, fontsize=18);	plt.ylabel(yla, fontsize=18); 	plt.legend()
            plt.show()
    
    def plot_3(self,xla = 'epoch times', yla = 'loss'):
        ran = np.arange(50000)
        if(IPLOT):
            fig,ax = plt.subplots()
            ax.plot(ran,self.loss_result,label = "training_loss")
            ax.plot(ran,self.loss_validation_result,label = "validation_loss")
            plt.xlabel(xla, fontsize=18)
            plt.ylabel(yla, fontsize=18)  
            plt.legend()
            plt.show()


    def optimize(self,algo,LR,method):
        if algo == 'GD':
            t_max = 50000
            dp = 0.0001
            p = np.random.uniform(0.5,1,size = NFIT)
            self.loss_result = []
            self.loss_validation_result = []
            if method == 'batch':
                for t in range(0,t_max):
                    dl_result = []
                    p_last_iteration = copy.deepcopy(p)
                    for i in range(len(p)):
                        p[i] = p[i] + dp
                        dl_result.append((self.loss(p,self.train_idx)-self.loss(p_last_iteration,self.train_idx))/dp) 
                    for i in range(len(dl_result)):
                        dl_result[i] = dl_result[i] *  LR
                    gd_p = dl_result
                    p = p_last_iteration - gd_p
                    self.loss_result.append(self.loss(p,self.train_idx))
                    self.loss_validation_result.append(self.loss(p,self.val_idx))
        
            if method == 'mini_batch':
                first_half_idx = self.train_idx[self.train_idx%2==1]
                second_half_idx = self.train_idx[self.train_idx%2!=1]
                for t in range(0,t_max):
                    dl_result_first_half = []
                    dl_result_second_half = []
                    p_last_iteration = copy.deepcopy(p)
                    for i in range(len(p)):
                        p[i] = p[i] + dp 
                        dl_result_first_half.append((self.loss(p,first_half_idx) - self.loss(p_last_iteration,first_half_idx))/dp)
                    for i in range(len(dl_result_first_half)):
                        dl_result_first_half[i] = dl_result_first_half[i] * LR
                    gd_p = dl_result_first_half
                    p_after_first_half = p_last_iteration - gd_p  


                    p = copy.deepcopy(p_after_first_half)  #need a new copy to modify
                    for i in range(len(p)):
                        p[i] = p[i] +  dp
                        dl_result_second_half.append((self.loss(p,second_half_idx) - self.loss(p_after_first_half,second_half_idx))/dp)
                    for i in range(len(dl_result_second_half)):
                        dl_result_second_half[i] = dl_result_second_half[i] * LR
                    gd_p = dl_result_second_half
                    p = p_after_first_half - gd_p
                    self.loss_result.append(self.loss(p,self.train_idx))
                    self.loss_validation_result.append(self.loss(p,self.val_idx))
            
            if method == 'stochastic':
                for t in range(0,t_max):
                    np.random.shuffle(self.train_idx)
                    for x in self.train_idx:
                        dl_list = []
                        loss = np.mean((self.model(self.X[x],p) - self.Y[x])**2 )
                        p_last_iteration = copy.deepcopy(p)

                        for i in range(len(p)):
                            p[i] = p[i] + dp 
                            dl_list.append((np.mean((self.model(self.X[x],p) - self.Y[x]))**2 - loss ) / dp)
                        
                        for  i in range(len(dl_list)):
                            dl_list[i] = dl_list[i] * LR
                        gd_p = dl_list
                        p = p_last_iteration - gd_p 

                    self.loss_result.append(self.loss(p,self.train_idx))
                    self.loss_validation_result.append(self.loss(p,self.val_idx))
                    # if t % 1000 == 0:
                    #     print(t)
                    #     print(dl_list)
                    #     print(p)
          

        if algo == 'GD_mom':
            t_max = 50000
            dp = 0.0001
            p = np.random.uniform(0.5,1,size = NFIT)
            self.loss_result = []
            self.loss_validation_result = []
            if method == 'batch':
                for t in range(0,t_max):
                    dl_result = []
                    p_last_iteration = copy.deepcopy(p)
                    for i in range(len(p)):
                        p[i] = p[i] + dp
                        dl_result.append((self.loss(p,self.train_idx)-self.loss(p_last_iteration,self.train_idx))/dp) 
                    for i in range(len(dl_result)):
                        dl_result[i] = dl_result[i] *  LR 
                    gd_p = dl_result
                    for j in range(len(p)):
                        p = p_last_iteration - gd_p 
                        p[i] = p[i] + dp * 0.03
                    self.loss_result.append(self.loss(p,self.train_idx))
                    self.loss_validation_result.append(self.loss(p,self.val_idx))

            if method == 'mini_batch':
                first_half_idx = self.train_idx[self.train_idx%2==1]
                second_half_idx = self.train_idx[self.train_idx%2!=1]
                for t in range(0,t_max):
                    dl_result_first_half = []
                    dl_result_second_half = []
                    p_last_iteration = copy.deepcopy(p)
                    for i in range(len(p)):
                        p[i] = p[i] + dp 
                        dl_result_first_half.append((self.loss(p,first_half_idx) - self.loss(p_last_iteration,first_half_idx))/dp)
                    for i in range(len(dl_result_first_half)):
                        dl_result_first_half[i] = dl_result_first_half[i] * LR
                    gd_p = dl_result_first_half
                    for j in range(len(p)):
                        p_after_first_half = p_last_iteration - gd_p  
                        p_after_first_half[i] = p_after_first_half[i] + dp * 0.03


                    p = copy.deepcopy(p_after_first_half)  #need a new copy to modify
                    for i in range(len(p)):
                        p[i] = p[i] +  dp
                        dl_result_second_half.append((self.loss(p,second_half_idx) - self.loss(p_after_first_half,second_half_idx))/dp)
                    for i in range(len(dl_result_second_half)):
                        dl_result_second_half[i] = dl_result_second_half[i] * LR
                    gd_p = dl_result_second_half

                    for j in range(len(p)):
                        p = p_after_first_half - gd_p
                        p[i] = p[i] + dp * 0.03
                    self.loss_result.append(self.loss(p,self.train_idx))
                    self.loss_validation_result.append(self.loss(p,self.val_idx))

            if method == 'stochastic':
                for t in range(0,t_max):
                    np.random.shuffle(self.train_idx)
                    for x in self.train_idx:
                        dl_list = []
                        loss = np.mean((self.model(self.X[x],p) - self.Y[x])**2 )
                        p_last_iteration = copy.deepcopy(p)

                        for i in range(len(p)):
                            p[i] = p[i] + dp 
                            dl_list.append((np.mean((self.model(self.X[x],p) - self.Y[x]))**2 - loss ) / dp)
                        
                        for  i in range(len(dl_list)):
                            dl_list[i] = dl_list[i] * LR
                        gd_p = dl_list
                        for j in range(len(p)):
                            p = p_last_iteration - gd_p 
                            p[i] = p[i] + dp*0.03

                    self.loss_result.append(self.loss(p,self.train_idx))
                    self.loss_validation_result.append(self.loss(p,self.val_idx))

        return p 
#------------------------
#MAIN 
#------------------------

# method = 'batch'
method = 'mini_batch'
# method = 'stochastic'
D=DataClass(INPUT_FILE)		#INITIALIZE DATA OBJECT 
D.report()					#BASIC DATA PRESCREENING

D.partition()				#SPLIT DATA
D.normalize()				#NORMALIZE
D.fit()
D.predict()
D.plot_1()					#PLOT DATA
D.plot_2()					#PLOT DATA
D.plot_3()

D.un_normalize()			#NORMALIZE
D.plot_1()					#PLOT DATA
D.plot_2()					#PLOT DATA
D.plot_3()

