import numpy as np
from code import imgProcess
import math
import pickle

class BayesModel:
    def __init__(self,train_image,train_label,dim):
        self.train_image=train_image
        self.train_label=train_label
        self.dim=dim
        self.hasModel=False
        self.PW=np.empty(10,dtype=np.float)
        self.PJW=np.empty(shape=(10, self.dim), dtype=np.float)

    def train_model(self,n=-1):
        NI={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}

        if len(self.train_image)!=len(self.train_label):
            print("Number of train_image is not equal to number of train_label !")
            return

        N=len(self.train_label)
        if n>-1:
            if n>N:
                print ("Do Not Have Enough Images To Train , N = %d ."%N)
            else:
                N=n
                print ("N = %d"%N)
        for lable in self.train_label:
            NI[int(lable)]+=1
        PNI={}
        for key in NI:
            PNI[key]=float(NI[key]/N)

        NJW=np.zeros(shape=(10,self.dim),dtype=np.int)
        PJW = np.zeros(shape=(10, self.dim), dtype=np.float)


        for i in range(N):
            cur_label=int(self.train_label[i])
            cur_raw_img=self.train_image[i]
            img_inLine=np.ndarray.flatten(cur_raw_img,order='C')
            for order in range(len(img_inLine)):
                NJW[cur_label][order]+=img_inLine[order]
            if i%500==0:
                print("Has trained %d pictures."%i)

        for i in range(10):
            for j in range(self.dim):
                PJW[i][j]=float((NJW[i][j]+1)/(NI[i]+2))

        # PXW=np.zeros(10,dtype=np.float)
        # for i in range(10):
        #     res=float(1)
        #     for j in range(self.dim):
        #         res=res*PJW[i][j]
        #     PXW[i]=res

        PW=np.zeros(10,dtype=np.float)
        for i in range(10):
            PW[i]=float(NI[i]/N)

        self.hasModel=True
        self.PW=PW
        self.PJW=PJW
        print("Training model finished .")

        return (PW,PJW)

    def predict(self,test_image,test_label):
        if self.hasModel==False:
            print("Model has not trained!")
            return

        N=len(test_image)
        correct=0
        for i in range(N):
            p={0:0.0,1:0.0,2:0.0,3:0.0,4:0.0,5:0.0,6:0.0,7:0.0,8:0.0,9:0.0}
            img=np.ndarray.flatten(test_image[i])
            for classification in range(10):
                res=math.log(self.PW[classification])
                for k in range(self.dim):
                    if img[k]==1:
                        res = res +math.log( self.PJW[classification][k])
                    else:
                        res = res+math.log  (1-self.PJW[classification][k])
                p[classification]=res
            sortted_p=sorted(p.items(), key=lambda x: x[1], reverse=True)
            # print("Picture_Num : %d , predict result is %d , real result is %d ."%(i,int(sortted_p[0][0]),int(test_label[i])))
            if int(sortted_p[0][0])==int(test_label[i]):
                correct+=1

        acc=correct/N
        return acc

    def saveModelToFile(self,filename):
        if self.hasModel==False:
            print("Model hasn't trained. CAN NOT SAVE.")
            return
        prefix="../model/"
        suffix=".pkl"

        f_pw=prefix+filename+"-pw"+suffix
        f_pjw=prefix+filename+"-pjw"+suffix

        with open(f_pw,'wb') as f__pw:
            with open(f_pjw,'wb') as f__pjw:
                picklestring = pickle.dump(self.PW, f__pw)
                picklestring = pickle.dump(self.PJW, f__pjw)
        print("Model has saved .")

        return

    def loadModelFromFile(self,filename):
        prefix="../model/"
        suffix=".pkl"

        f_pw=prefix+filename+"-pw"+suffix
        f_pjw=prefix+filename+"-pjw"+suffix

        with open(f_pw,'rb') as f__pw:
            with open(f_pjw,'rb') as f__pjw:

                self.PW=pickle.load(f__pw)
                self.PJW=pickle.load(f__pjw)

        self.hasModel=True
        print("Model has loaded .")
        return










