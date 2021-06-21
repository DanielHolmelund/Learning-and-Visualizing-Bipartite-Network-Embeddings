import numpy as np
import matplotlib.pyplot as plt
import csv
import os
os.chdir("results/binary/")

train_loss = []
test_loss = []
fprs = []
tprs = []
tpr_org = []
base_fpr = np.linspace(0, 1, 101)


for i in range(3):
    train_loss.append(np.loadtxt(f"cum_loss_train_5000_{i}.txt").tolist())
    test_loss.append(np.loadtxt(f"cum_loss_test_5000_{i}.txt").tolist())
    fprs.append(np.loadtxt(f"tpr_5000_{i}.txt").tolist())
    tpr_org.append(np.loadtxt(f"fpr_5000_{i}.txt").tolist())


for i in range(3):
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.plot(fprs[i], tpr_org[i], 'b', alpha=0.15)
    tpr = np.interp(base_fpr, fprs[i], tpr_org[i])
    tpr[0] = 0.0
    tprs.append(tpr)

iterations = len(train_loss[0])

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis=0)
std = tprs.std(axis=0)

# USing standard deviation as error bars
tprs_upper = np.minimum(mean_tprs + std, 1)
tprs_lower = mean_tprs - std

plt.plot(base_fpr, mean_tprs, 'b', label='Mean ROC-curve')
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

plt.plot([0, 1], [0, 1], 'r--', label='Random classifier')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.axes().set_aspect('equal', 'datalim')
plt.grid()
plt.legend()
plt.show()
plt.savefig('Average_ROC_curve.png')

# Plotting the average loss based on the cross validation
train_loss = np.array(train_loss)
test_loss = np.array(test_loss)
mean_train_loss = train_loss.mean(axis=0)
std_train_loss = train_loss.std(axis=0)
mean_train_loss_upr = mean_train_loss + std_train_loss
mean_train_loss_lwr = mean_train_loss - std_train_loss

mean_test_loss = test_loss.mean(axis=0)
std_test_loss = test_loss.std(axis=0)
mean_test_loss_upr = mean_test_loss + std_test_loss
mean_test_loss_lwr = mean_test_loss - std_test_loss

plt.plot(np.arange(iterations), mean_train_loss, 'b', label='Mean training loss')
plt.fill_between(np.arange(iterations), mean_train_loss_lwr, mean_train_loss_upr, color='b', alpha=0.3)
plt.plot(np.arange(iterations), mean_test_loss, 'r', label='Mean test loss')
plt.fill_between(np.arange(iterations), mean_test_loss_lwr, mean_test_loss_upr, color='r', alpha=0.3)
plt.xlim([0, iterations])
plt.ylabel('Loss')
plt.xlabel('Iterations')
plt.grid()
plt.legend()
plt.show()
plt.savefig('Average_loss.png')
