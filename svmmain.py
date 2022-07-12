
from sklearn.linear_model import LogisticRegression
import numpy as np
from math import exp
np.random.seed(0)

'''
Making synthetic data
First generate W* and then generate all the different clusters
'''


features = []

while True:
    x_star = (np.random.random(7)-0.5)*6 #generate X_star such that the mean A doesn't have any absurdly large numbrs

    a_mean = (1/x_star)/7
    if all(a_mean<5):
        break

print("X* value: " + str(x_star))
print("Mean a value: " + str(a_mean))


for noise in [i*20 for i in range(1, 6)]:
    features.append(np.random.normal(a_mean, noise, (50, 7)))
    
features = np.asarray(features)


model = LogisticRegression(fit_intercept=False)

labels = np.zeros((5, 50))

for i, task in enumerate(features):
    for j, sample in enumerate(task):
        
        P = 1/(1-exp(np.dot(sample, x_star)))
        
        labels[i][j] = 1 if P>0.5 else 0

all_data  = features.reshape(250, 7)
all_labels = labels.reshape(250)
model_star = LogisticRegression()
model_star.fit(all_data, all_labels)
x_star = model_star.coef_

for ind, task_num in enumerate([0, 2]):
    if ind>0: vect = model.coef_
    model.fit(features[task_num], labels[task_num])
    
    print(np.linalg.norm(model.coef_ - x_star))
    if ind>0: print(np.linalg.norm(vect-model.coef_))
    
    # print(model1.predict(features[task_num]))
    # print(model1.decision_function(features[task_num]))
    #print([1/(1+exp(np.dot(x_star, f))) for f in features[task_num]])

    













        