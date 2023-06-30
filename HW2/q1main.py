import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

####Change direction####
dir_name = r"C:\Users\Oguz\Desktop\Bilkent\3.2\CS464\hw2\afhq_cat"

print("Please wait, it can take few minutes to run completely")
pics=[]
for i in os.listdir(dir_name):
    pics.append(Image.open(os.path.join(dir_name,i)).resize((64,64), Image.BILINEAR))

imgs=[]
for j in pics:
    imgs.append(np.array(j,dtype=float).reshape(-1,3))

X=np.array(imgs)

X_0=X[:,:,0]
X_1=X[:,:,1]
X_2=X[:,:,2]

# # Question 1.1

def pca(matrix):
    matrix_mean=matrix-np.mean(matrix,axis=0)
    cov=np.cov(matrix_mean, rowvar = False)
    value, vector = np.linalg.eig(cov)
    sort_index = np.argsort(value)[::-1]
    sort_eigenval = value[sort_index]
    sort_eigenvec = vector[:,sort_index]
    return sort_eigenval, sort_eigenvec

val_0,vec_0=pca(X_0)
val_1,vec_1=pca(X_1)
val_2,vec_2=pca(X_2)

def pve(sort_val):
    n=len(sort_val)
    pve = sort_val*100/np.sum(sort_val)
    pve_sum=pve.dot(np.triu(np.ones((n,n))))
    return pve, pve_sum

pve_R,pve_sum_R=pve(val_0)
pve_G,pve_sum_G=pve(val_1)
pve_B,pve_sum_B=pve(val_2)

pve_table=pd.DataFrame(np.array([pve_R,pve_sum_R,pve_G,pve_sum_G,pve_B,pve_sum_B]).T \
,index=np.arange(1,len(pve_G)+1),columns= \
["PVE Red","PVE Red Sums","PVE Green","PVE Green Sums","Pve Blue","PVE Blue Sums"])
#pve_table[:10]
print(pve_table[:10])

fig,ax=plt.subplots(1,3,figsize=(30,6))

ax[0].plot(pve_R[:50],"r")
ax[0].set_ylabel("PVE of Red Principal Components")
ax[0].set_xlabel("Number of Principal Components")
ax[0].set_title("Proportion of Variance Explained for \
First 50 Red Principal Components")

ax[1].plot(pve_G[:50],"g")
ax[1].set_ylabel("PVE of Green Principal Components")
ax[1].set_xlabel("Number of Principal Components")
ax[1].set_title("Proportion of Variance Explained for \
First 50 Green Principal Components")

ax[2].plot(pve_B[:50],"b")
ax[2].set_ylabel("PVE of Blue Principal Components")
ax[2].set_xlabel("Number of Principal Components")
ax[2].set_title("Proportion of Variance Explained for \
First 50 Blue Principal Components")

fig.suptitle("Proportion of Variance Explained for \
 First 50 Principal Components for Each Color")
fig.show()

def threshold(th,pve):
    th_mat=np.argwhere(pve>th)
    return (th_mat[0][0]+1, pve[th_mat[0][0]])

th_R=threshold(70,pve_sum_R)
th_G=threshold(70,pve_sum_G)
th_B=threshold(70,pve_sum_B)

print("While the sum of first {} principal component PVE is\
 {:.3f}%, sum of first {} principal component PVE is {:.3f}% \
for red color.".format(th_R[0]-1,pve_sum_R[th_R[0]-2],th_R[0],th_R[1]))

print("While the sum of first {} principal component PVE is \
{:.3f}%, sum of first {} principal component PVE is {:.3f}% \
for green color. ".format(th_G[0]-1,pve_sum_G[th_G[0]-2],th_G[0],th_G[1]))

print("While the sum of first {} principal component PVE is \
{:.3f}%, sum of first {} principal component PVE is {:.3f}% \
for blue color.".format(th_B[0]-1,pve_sum_B[th_B[0]-2],th_B[0],th_B[1]))

fig,ax=plt.subplots(1,3,figsize=(30,6))

ax[0].plot(pve_sum_R,"r")
ax[0].annotate("({},{:.3f})".format(th_R[0],th_R[1]),th_R,\
 xytext=(th_R[0]+100,th_R[1]+5),arrowprops=dict(arrowstyle="->") )
ax[0].set_ylabel("Sum of PVE of Red Principal Components")
ax[0].set_xlabel("Number of Principal Components")
ax[0].set_title("The Sum of Proportion of Variance \
Explained for Red Principal Components")

ax[1].plot(pve_sum_G,"g")
ax[1].annotate("({},{:.3f})".format(th_G[0],th_G[1]),th_G,\
 xytext=(th_G[0]+100,th_G[1]+5),arrowprops=dict(arrowstyle="->") )
ax[1].set_ylabel("Sum of PVE of Green Principal Components")
ax[1].set_xlabel("Number of Principal Components")
ax[1].set_title("The Sum of Proportion of Variance \
Explained for Green Principal Components")

ax[2].plot(pve_sum_B,"b")
ax[2].annotate("({},{:.3f})".format(th_B[0],th_B[1]),th_B,\
 xytext=(th_B[0]+100,th_B[1]+5),arrowprops=dict(arrowstyle="->") )
ax[2].set_ylabel("Sum of PVE of Blue Principal Components")
ax[2].set_xlabel("Number of Principal Components")
ax[2].set_title("The Sum of Proportion of Variance \
Explained for Blue Principal Components")

fig.suptitle("The Sum of Proportion of Variance \
Explained for All Principal Components")
fig.show()

# # Question 1.2

normalization = lambda x:(x-x.min()) / (x.max()-x.min())

def first_10(mat):
    _10_pca=mat[:,:10]
    reshaped_10_pca=np.transpose(_10_pca, axes= (1,0)).reshape(10,64, 64)
    return reshaped_10_pca

reshaped_R=normalization(first_10(vec_0))
reshaped_G=normalization(first_10(vec_1))
reshaped_B=normalization(first_10(vec_2))

ten_img=np.array([reshaped_R,reshaped_G,reshaped_B])
ten_img=np.transpose(ten_img,axes=(1,2,3,0))

row = 2
col = 5
plt.figure(figsize=(18,7))
for q in range(10):
    plt.subplot(row,col,q+1)
    plt.imshow(ten_img[q])
    plt.title("RGB Image of Eigenvector {}".format(q+1))
plt.suptitle("RGB Images of First 10 Eigenvectors")

# # Question 1.3

def reconstruct(data, eigvec):
    mean=np.mean(data,axis=0)
    data=data-mean
    project = np.matmul(data,eigvec.dot(eigvec.T)) + mean
    return project

k_list= [1, 50, 250, 500, 1000, 4096]

sec_img=Image.open(os.path.join(dir_name,"flickr_cat_000003.jpg"))\
.resize((64,64), Image.BILINEAR)
sec_img=np.array(sec_img,dtype=np.float32).reshape(-1,3)

R_list=[]
G_list=[]
B_list=[]
for k in k_list:
    R_list.append(reconstruct(sec_img[:,0], vec_0[:,:k]))
    G_list.append(reconstruct(sec_img[:,1], vec_1[:,:k]))
    B_list.append(reconstruct(sec_img[:,2], vec_2[:,:k]))

rec=np.array([R_list,G_list,B_list])
rec=np.transpose(rec,axes=(1,2,0)).reshape(6,64,64,3)

plt.figure(figsize=(18,10))
for l in range(6):
    plt.subplot(2,3,l+1)
    plt.imshow(normalization(rec[l]).reshape(64,64,3))
    plt.title("Reconsturction From First {} \
Principal Components".format(k_list[l]))
plt.suptitle("Reconstruction for Different \
First Principal Components Sizes")

errors=[]
for i in range(len(k_list)):
    errors.append(np.square(sec_img.reshape(64,64,3)-rec[i]).sum()/(64*64*3))

plt.figure(figsize=(16,8))
plt.plot(k_list,errors,"bo-")
plt.xlabel("Number of First Principal Components")
plt.ylabel("Mean Squared Error")
plt.title("MSE of Wanted First Principal Components")

plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.imshow(normalization(rec[5]).reshape(64,64,3))
plt.title("Best Reconstruction")
plt.subplot(1,2,2)
plt.imshow((normalization(sec_img).reshape(64,64,3)))
plt.title("Original Image")
plt.suptitle("Comparison of Best Reconstruction and Original Image")
plt.show()