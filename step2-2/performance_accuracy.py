import PIL.Image as pilimg
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score, accuracy_score
from matplotlib import pyplot as plt
import os

def segmentation(input_image, height, width, threshold=0.5):
    output_image = input_image
    for i in range(height):
        for j in range(width):
            if output_image[i][j] <= threshold :
                output_image[i][j] = 1
            else:
                output_image[i][j] = 0
    return output_image

file_list1 = []
file_dir1 = "output"
file_list1 = os.listdir(file_dir1)
path1 = []
temp = []
pred = []
for i, j in enumerate(file_list1):
    path1.append("output/" + file_list1[i])    
for i, j in enumerate(path1):
    img1 = str(path1[i])   
    im_pred = pilimg.open(img1)
    pix_pred = np.array(im_pred)
    pix_pred = pix_pred/255
    pix_temp = pix_pred
    pix_temp = np.ravel(pix_temp, order='C')
    temp.extend(pix_temp)

    pix_pred = segmentation(pix_pred, 256, 256)
    pix_pred = pix_pred.astype(np.int64)
    pix_pred = np.ravel(pix_pred, order='C')
    pred.extend(pix_pred)

#print(np.array(temp).shape)
#print(np.array(pred).shape)

file_list2 = []
file_dir2 = "label"
file_list2 = os.listdir(file_dir2)
path2 = []
actu = []
for i, j in enumerate(file_list2):
    path2.append("label/" + file_list2[i])    
for i, j in enumerate(path2):
    img2 = str(path2[i])   
    im_actu = pilimg.open(img2)
    pix_actu = np.array(im_actu)
    pix_actu = pix_actu/255
    pix_actu = segmentation(pix_actu, 256, 256)
    pix_actu = pix_actu.astype(np.int64)
    pix_actu = np.ravel(pix_actu, order='C')
    actu.extend(pix_actu)

#print(np.array(actu).shape)

print(confusion_matrix(actu,pred))
print("precision score:", precision_score(actu, pred))
print("recall score", recall_score(actu, pred))

fpr, tpr, thresholds = roc_curve(actu, temp)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
plt.savefig('fig1.png', dpi=600)
plt.show()

print("roc_auc:", roc_auc_score(actu, pred))
print("accuracy score:", accuracy_score(actu, pred))