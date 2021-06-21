import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir("results/binary")
train_loss, test_loss = [], []
for i in range(3):
    train_loss.append(np.loadtxt(f"cum_loss_train_5000_{i}.txt",unpack=False))
    test_loss.append(np.loadtxt(f"cum_loss_test_5000_{i}.txt",unpack=False))


iterations = np.shape(train_loss)[1]

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
plt.savefig('Average_binary_loss.png')
