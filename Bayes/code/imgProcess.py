import numpy as np


class ImgProcess:
    def __init__(self):
        self.num_rows=28
        self.num_cols=28

    def gray2bw(self,img,T=80):
        res =np.zeros(shape=(self.num_rows,self.num_cols))
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if img[i][j]>T:
                    res[i][j]=255
                else:
                    res[i][j]=0
        return res

    def gray2bw_standard(self,img,T=80):
        res =np.zeros(shape=(self.num_rows,self.num_cols))
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if img[i][j]>T:
                    res[i][j]=1
                else:
                    res[i][j]=0
        return res

    def maxPooling(self,img,r_size,c_size=-1):
        if c_size==-1:
            c_size=r_size
        if self.num_rows%r_size!=0 or self.num_cols%c_size!=0:
            print("Warning: MaxPooling will lose some data!")
        res_r_size=self.num_rows/r_size
        res_c_size=self.num_cols/c_size
        res=np.zeros(shape=(int(self.num_rows/r_size),int(self.num_cols/c_size)))
        for i in range(int(res_r_size)):
            for j in range(int(res_c_size)):
                value_list=[]
                max_value=0
                for k in range(r_size):
                    for l in range(c_size):
                        value_list.append(img[r_size*i+k][c_size*j+l])
                        max_value=np.amax(value_list)
                res[i][j]=max_value

        return res

    def minPooling(self,img,r_size,c_size=-1):
        if c_size==-1:
            c_size=r_size
        if self.num_rows%r_size!=0 or self.num_cols%c_size!=0:
            print("Warning: MinPooling will lose some data!")
        res_r_size=self.num_rows/r_size
        res_c_size=self.num_cols/c_size
        res=np.zeros(shape=(int(self.num_rows/r_size),int(self.num_cols/c_size)))
        for i in range(int(res_r_size)):
            for j in range(int(res_c_size)):
                value_list=[]
                max_value=0
                for k in range(r_size):
                    for l in range(c_size):
                        value_list.append(img[r_size*i+k][c_size*j+l])
                        max_value=np.amin(value_list)
                res[i][j]=max_value

        return res

    def avgPooling(self,img,r_size,c_size=-1):
        if c_size==-1:
            c_size=r_size
        if self.num_rows%r_size!=0 or self.num_cols%c_size!=0:
            print("Warning: AvgPooling will lose some data!")
        res_r_size=self.num_rows/r_size
        res_c_size=self.num_cols/c_size
        res=np.zeros(shape=(int(self.num_rows/r_size),int(self.num_cols/c_size)))
        for i in range(int(res_r_size)):
            for j in range(int(res_c_size)):
                value_list=[]
                max_value=0
                for k in range(r_size):
                    for l in range(c_size):
                        value_list.append(img[r_size*i+k][c_size*j+l])
                        max_value=np.average(value_list)
                res[i][j]=max_value

        return res

