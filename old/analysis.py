import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import hamming_loss, f1_score, precision_recall_fscore_support

# Load true and predicted
trueLabels = np.load('testLabels.npy')
predLabels = np.load('predicted.npy') 


# Flatten down into segmentations 
trueLabels = np.argmax(trueLabels, axis=3) 
predLabels = np.argmax(predLabels, axis=3) 

mask = (trueLabels > 0)


TP = np.sum( np.logical_and (mask, np.equal(trueLabels, predLabels)) ) / float(np.sum(mask)) 
print("Masked accuracy rate: ", TP)

maskF = mask.flatten()
trueF = (trueLabels.flatten())[maskF]
predF = (predLabels.flatten())[maskF]

ham = hamming_loss(trueF, predF)
print(ham) 

prfs = precision_recall_fscore_support(trueF, predF, average='micro')
print(prfs)

plt.subplot(1,2,1)
plt.imshow(predLabels[14,:,:])
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(trueLabels[14,:,:])
plt.axis('off')
plt.show()
