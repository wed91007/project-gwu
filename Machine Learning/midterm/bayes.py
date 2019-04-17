# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 13:35:33 2019

@author: Administrator
"""

#readfile
data=list()
f=open(r"F:/project-gwu/Machine Learning/midterm/datanbc.txt")
line=f.readline()
for line in f.readlines():
    line=line.strip()
    data.append(line)
f.close()
count=len(data)
label=list()
Data=[[0 for i in range(4)]for j in range(count)]

for i in range(0,count):
     a = data[i].split()
     label.append(a[5])
     a=a[1:5]
     for j in range(4):
         Data[i][j]=a[j]
index_yn=0
index_yy=0
for i in range(count):
    if label[i]=="no":
        index_yn +=1
    else: 
        index_yy +=1
#store probability
matx1=[[float(0) for i in range(2)]for j in range(3)]
matx2=[[float(0) for i in range(2)]for j in range(3)]
matx3=[[float(0) for i in range(2)]for j in range(2)]
matx4=[[float(0) for i in range(2)]for j in range(2)]
px1=[[float(0) for i in range(2)]for j in range(3)]
px2=[[float(0) for i in range(2)]for j in range(3)]
px3=[[float(0) for i in range(2)]for j in range(2)]
px4=[[float(0) for i in range(2)]for j in range(2)]
#pmat=[[[float(0) for i in range(2)]for j in range(4)]for k in range(count)]
#Compute Prior
for i in range(count):
    if Data[i][0]=="youth":
        if label[i]=="yes":
            matx1[0][0] +=1
        else:
            matx1[0][1] +=1
    if Data[i][0]=="middle-aged":
        if label[i]=="yes":
            matx1[1][0] +=1
        else:
            matx1[1][1] +=1
    if Data[i][0]=="senior":
        if label[i]=="yes":
            matx1[2][0] +=1
        else:
            matx1[2][1] +=1
    
    if Data[i][1]=="low":
        if label[i]=="yes":
            matx2[0][0] +=1
        else:
            matx2[0][1] +=1
    if Data[i][1]=="medium":
        if label[i]=="yes":
            matx2[1][0] +=1
        else:
            matx2[1][1] +=1
    if Data[i][1]=="high":
        if label[i]=="yes":
            matx2[2][0] +=1
        else:
            matx2[2][1] +=1

    if Data[i][2]=="yes":
        if label[i]=="yes":
            matx3[0][0] +=1
        else:
            matx3[0][1] +=1
    if Data[i][2]=="no":
        if label[i]=="yes":
            matx3[1][0] +=1
        else:
            matx3[1][1] +=1        
        
    if Data[i][3]=="fair":
        if label[i]=="yes":
            matx4[0][0] +=1
        else:
            matx4[0][1] +=1
    if Data[i][3]=="excellent":
        if label[i]=="yes":
            matx4[1][0] +=1
        else:
            matx4[1][1] +=1
            
for i in range(3):
    px1[i][0]=matx1[i][0]/index_yy
    px1[i][1]=matx1[i][1]/index_yn
    px2[i][0]=matx2[i][0]/index_yy
    px2[i][1]=matx2[i][1]/index_yn              
for i in range(2):
    px3[i][0]=matx3[i][0]/index_yy
    px3[i][1]=matx3[i][1]/index_yn            
    px4[i][0]=matx4[i][0]/index_yy
    px4[i][1]=matx4[i][1]/index_yn 
 
if __name__ == '__main__':    

    test=['youth','low','yes','fair']
    resulty=px1[0][0]*px2[0][0]*px3[0][0]*px4[0][0]
    resultn=px1[0][1]*px2[0][1]*px3[0][1]*px4[0][1]
    print resulty, resultn
