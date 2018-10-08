import pickle
import numpy as np
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


with open('./dumps/MultiColProcessor.pkl', 'rb') as handle:
    MultiColProcessor = pickle.load(handle)
##
with open('./dumps/category_mapper.json') as handle:
    category_encoder = json.load(handle)

y_test = np.argmax(y_true, axis=1)
y_pred = np.argmax(logits_pred, axis=1)

category_encoder['num_to_cat']['17']

cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()