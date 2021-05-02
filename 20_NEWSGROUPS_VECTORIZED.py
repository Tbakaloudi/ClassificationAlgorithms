#!/usr/bin/env python
# coding: utf-8

# # 20 newsgroups vectorized

# In[1]:


from sklearn.datasets import fetch_20newsgroups_vectorized
import numpy as np
newsgroup = fetch_20newsgroups_vectorized(normalize = False)
newsgroup_std = fetch_20newsgroups_vectorized(normalize = True)
# All features
X = newsgroup.data
X_std = newsgroup_std.data
# LFW people classes
y = newsgroup.target
y_std = newsgroup_std.target
print('Class labels:', np.unique(y))


# In[2]:


#Splitting data into 70% training and 30% test data
from sklearn.model_selection import train_test_split

# Training and Test sets - all features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify = y)
X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X_std, y_std, test_size = 0.3, random_state = 1, stratify = y)


# In[3]:


print('Labels count in y:', np.bincount(y))
print('Labels count in y_train:', np.bincount(y_train))
print('Labels count in y_test:', np.bincount(y_test))


# ## Logistic Regression
# ### Without standardization

# In[4]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter = 3000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)


# In[5]:


# Evaluation Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import hamming_loss

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
mse = mean_squared_error(y_test, y_pred)
hl = hamming_loss(y_test, y_pred)

print('Accuracy', accuracy)
print('F1', f1)
print('MSE', mse)
print('Hamming Loss', hl)

acc1_ns = accuracy
f11_ns = f1
mse1_ns = mse
hl1_ns = hl


#
# ###  With Standardization

# In[6]:


lr = LogisticRegression(max_iter = 3000)
lr.fit(X_train_std, y_train)
y_pred = lr.predict(X_test_std)


# In[7]:


# Evaluation Metrics

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
mse = mean_squared_error(y_test, y_pred)
hl = hamming_loss(y_test, y_pred)

print('Accuracy', accuracy)
print('F1', f1)
print('MSE', mse)
print('Hamming Loss', hl)

acc1_s = accuracy
f11_s = f1
mse1_s = mse
hl1_s = hl


# # Perceptron
# ### Without standardization

# In[8]:


# Perceptron model training
from sklearn.linear_model import Perceptron

# ppn: Perceptron trained by taking into account all features
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train, y_train)


# In[9]:


# Misclassifications
y_pred = ppn.predict(X_test)
print('Misclassified examples (All Features): %d' % (y_test != y_pred).sum())


# In[10]:


# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
mse = mean_squared_error(y_test, y_pred)
hl = hamming_loss(y_test, y_pred)

print('Accuracy', accuracy)
print('F1', f1)
print('MSE', mse)
print('Hamming Loss', hl)

acc2_ns = accuracy
f12_ns = f1
mse2_ns = mse
hl2_ns = hl


# ### With standardization

# In[11]:


# Perceptron model training
# ppn: Perceptron trained by taking into account all features
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)


# In[12]:


# Misclassifications
y_pred = ppn.predict(X_test_std)
print('Misclassified examples (All Features): %d' % (y_test != y_pred).sum())


# In[13]:


# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
mse = mean_squared_error(y_test, y_pred)
hl = hamming_loss(y_test, y_pred)

print('Accuracy', accuracy)
print('F1', f1)
print('MSE', mse)
print('Hamming Loss', hl)

acc2_s = accuracy
f12_s = f1
mse2_s = mse
hl2_s = hl


# ## SVM
# ### Without standardization

# In[14]:


from sklearn.svm import SVC

# svm: SVM model trained by taking into account all features
# RBF kernel
svm = SVC(kernel='rbf', C=1.0, random_state=1)
svm.fit(X_train, y_train)


# In[15]:


# Misclassifications
y_pred = svm.predict(X_test)
print('Misclassified examples (All Features): %d' % (y_test != y_pred).sum())


# In[16]:


# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
mse = mean_squared_error(y_test, y_pred)
hl = hamming_loss(y_test, y_pred)

print('Accuracy', accuracy)
print('F1', f1)
print('MSE', mse)
print('Hamming Loss', hl)

acc3_ns_rbf = accuracy
f13_ns_rbf = f1
mse3_ns_rbf = mse
hl3_ns_rbf = hl


# In[17]:


# RBF linear
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train, y_train)


# In[18]:


# Misclassifications
y_pred = svm.predict(X_test)
print('Misclassified examples (All Features): %d' % (y_test != y_pred).sum())


# In[19]:


# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
mse = mean_squared_error(y_test, y_pred)
hl = hamming_loss(y_test, y_pred)

print('Accuracy', accuracy)
print('F1', f1)
print('MSE', mse)
print('Hamming Loss', hl)

acc3_ns_linear = accuracy
f13_ns_linear = f1
mse3_ns_linear = mse
hl3_ns_linear = hl


# ### With standardization

# In[20]:


# With standardization

# svm: SVM model trained by taking into account all features
# RBF kernel
svm = SVC(kernel='rbf', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)


# In[21]:


# Misclassifications
y_pred = svm.predict(X_test_std)
print('Misclassified examples (All Features): %d' % (y_test != y_pred).sum())


# In[22]:


# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
mse = mean_squared_error(y_test, y_pred)
hl = hamming_loss(y_test, y_pred)

print('Accuracy', accuracy)
print('F1', f1)
print('MSE', mse)
print('Hamming Loss', hl)

acc3_s_rbf = accuracy
f13_s_rbf = f1
mse3_s_rbf = mse
hl3_s_rbf = hl


# In[23]:


# Linear kernel
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)


# In[24]:


# Misclassifications
y_pred = svm.predict(X_test_std)
print('Misclassified examples (All Features): %d' % (y_test != y_pred).sum())


# In[25]:


# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
mse = mean_squared_error(y_test, y_pred)
hl = hamming_loss(y_test, y_pred)

print('Accuracy', accuracy)
print('F1', f1)
print('MSE', mse)
print('Hamming Loss', hl)

acc3_s_linear = accuracy
f13_s_linear = f1
mse3_s_linear = mse
hl3_s_linear = hl


# # Decision Tree
# ## Without standardization

# In[26]:


from sklearn.tree import DecisionTreeClassifier

# Build a decision tree for classification tasks
# citerion:  The function to measure the quality of a split.
# max_depth: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or
#            until all leaves contain less than min_samples_split samples.
# min_samples_split: The minimum number of samples required to split an internal node.
# min_samples_leaf:  The minimum number of samples required to to be at a leaf node.
tree_model = DecisionTreeClassifier(criterion = 'gini', max_depth = 4, random_state = 1)
tree_model.fit(X_train, y_train)


# In[27]:


# Misclassifications
y_pred = tree_model.predict(X_test)
print('Misclassified examples (All Features): %d' % (y_test != y_pred).sum())


# In[28]:


# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
mse = mean_squared_error(y_test, y_pred)
hl = hamming_loss(y_test, y_pred)

print('Accuracy', accuracy)
print('F1', f1)
print('MSE', mse)
print('Hamming Loss', hl)

acc4_ns = accuracy
f14_ns = f1
mse4_ns = mse
hl4_ns = hl


# ## With standardization

# In[29]:


# Build a decision tree for classification tasks
# citerion:  The function to measure the quality of a split.
# max_depth: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or
#            until all leaves contain less than min_samples_split samples.
# min_samples_split: The minimum number of samples required to split an internal node.
# min_samples_leaf:  The minimum number of samples required to to be at a leaf node.
tree_model = DecisionTreeClassifier(criterion = 'gini', max_depth = 4, random_state = 1)
tree_model.fit(X_train_std, y_train)


# In[30]:


# Misclassifications
y_pred = tree_model.predict(X_test_std)
print('Misclassified examples (All Features): %d' % (y_test != y_pred).sum())


# In[31]:


# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
mse = mean_squared_error(y_test, y_pred)
hl = hamming_loss(y_test, y_pred)

print('Accuracy', accuracy)
print('F1', f1)
print('MSE', mse)
print('Hamming Loss', hl)

acc4_s = accuracy
f14_s = f1
mse4_s = mse
hl4_s = hl


# # Random Forests
# ## Without standardization

# In[32]:


from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest comprised of 25 decision trees
# citerion:  The function to measure the quality of a split.
# max_depth: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or
#            until all leaves contain less than min_samples_split samples.
# min_samples_split: The minimum number of samples required to split an internal node.
# min_samples_leaf:  The minimum number of samples required to to be at a leaf node.
forest = RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)


# In[33]:


# Misclassifications
y_pred = forest.predict(X_test)
print('Misclassified examples (All Features): %d' % (y_test != y_pred).sum())


# In[34]:


# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
mse = mean_squared_error(y_test, y_pred)
hl = hamming_loss(y_test, y_pred)

print('Accuracy', accuracy)
print('F1', f1)
print('MSE', mse)
print('Hamming Loss', hl)

acc5_ns = accuracy
f15_ns = f1
mse5_ns = mse
hl5_ns = hl


# ## With standardization

# In[35]:


forest = RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1, n_jobs=2)
forest.fit(X_train_std, y_train)


# In[36]:


# Misclassifications
y_pred = forest.predict(X_test_std)
print('Misclassified examples (All Features): %d' % (y_test != y_pred).sum())


# In[37]:


# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
mse = mean_squared_error(y_test, y_pred)
hl = hamming_loss(y_test, y_pred)

print('Accuracy', accuracy)
print('F1', f1)
print('MSE', mse)
print('Hamming Loss', hl)

acc5_s = accuracy
f15_s = f1
mse5_s = mse
hl5_s = hl


# # Feed-forward neural networks with 2 hidden layers
# ## Without standardization

# In[38]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
mlp.fit(X_train, y_train)


# In[39]:


# Misclassifications
y_pred = forest.predict(X_test)
print('Misclassified examples (All Features): %d' % (y_test != y_pred).sum())


# In[40]:


# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
mse = mean_squared_error(y_test, y_pred)
hl = hamming_loss(y_test, y_pred)

print('Accuracy', accuracy)
print('F1', f1)
print('MSE', mse)
print('Hamming Loss', hl)

acc6_ns = accuracy
f16_ns = f1
mse6_ns = mse
hl6_ns = hl


# ## With standardization

# In[41]:


mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
mlp.fit(X_train_std, y_train)


# In[42]:


# Misclassifications
y_pred = forest.predict(X_test_std)
print('Misclassified examples (All Features): %d' % (y_test != y_pred).sum())


# In[43]:


# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
mse = mean_squared_error(y_test, y_pred)
hl = hamming_loss(y_test, y_pred)

print('Accuracy', accuracy)
print('F1', f1)
print('MSE', mse)
print('Hamming Loss', hl)

acc6_s = accuracy
f16_s = f1
mse6_s = mse
hl6_s = hl


# #  Plots

# In[44]:


import matplotlib.pyplot as plt

acc_without_plot = [acc1_ns, acc2_ns, acc3_ns_rbf, acc3_ns_linear, acc4_ns, acc5_ns, acc6_ns]
objects = ('Logistic regression', 'Perceptron', 'SVM rbf','SVM linear', 'Decision Tree', 'Random Forests', 'FFNN')
y_pos = np.arange(len(objects))

plt.barh(y_pos, acc_without_plot, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('Accuracy')
plt.title('20 newsgroups vectorized: Accuracy without Standardization')


# In[45]:


acc_with_plot= [acc1_s, acc2_s, acc3_s_rbf, acc3_s_linear, acc4_s, acc5_s, acc6_s]
objects = ('Logistic regression', 'Perceptron', 'SVM rbf','SVM linear', 'Decision Tree', 'Random Forests', 'FFNN')
y_pos = np.arange(len(objects))

plt.barh(y_pos, acc_with_plot, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('Accuracy')
plt.title('20 newsgroups vectorized: Accuracy with Standardization')


# In[46]:


f1_without_plot = [f11_ns, f12_ns, f13_ns_rbf, f13_ns_linear, f14_ns, f15_ns, f16_ns]
objects = ('Logistic regression', 'Perceptron', 'SVM rbf','SVM linear', 'Decision Tree', 'Random Forests', 'FFNN')
y_pos = np.arange(len(objects))

plt.barh(y_pos, f1_without_plot, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('F1')
plt.title('20 newsgroups vectorized: F1 without standardization')


# In[47]:


f1_with_plot = [f11_s, f12_s, f13_s_rbf, f13_s_linear, f14_s, f15_s, f16_s]
objects = ('Logistic regression', 'Perceptron', 'SVM rbf','SVM linear', 'Decision Tree', 'Random Forests', 'FFNN')
y_pos = np.arange(len(objects))

plt.barh(y_pos, f1_with_plot, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('F1')
plt.title('20 newsgroups vectorized: F1 with standardization')


# In[48]:


mse_without_plot = [mse1_ns, mse2_ns, mse3_ns_rbf, mse3_ns_linear, mse4_ns, mse5_ns, mse6_ns]
objects = ('Logistic regression', 'Perceptron', 'SVM rbf','SVM linear', 'Decision Tree', 'Random Forests', 'FFNN')
y_pos = np.arange(len(objects))

plt.barh(y_pos, mse_without_plot, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('MSE')
plt.title('20 newsgroups vectorized: MSE without standardization')


# In[49]:


mse_with_plot = [mse1_s, mse2_s, mse3_s_rbf, mse3_s_linear, mse4_s, mse5_s, mse6_s]
objects = ('Logistic regression', 'Perceptron', 'SVM rbf','SVM linear', 'Decision Tree', 'Random Forests', 'FFNN')
y_pos = np.arange(len(objects))

plt.barh(y_pos, mse_with_plot, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('MSE')
plt.title('20 newsgroups vectorized: MSE with standardization')


# In[54]:


hl_without_plot = [hl1_ns, hl2_ns, hl3_ns_rbf, hl3_ns_linear, hl4_ns, hl5_ns, hl6_ns]
objects = ('Logistic regression', 'Perceptron', 'SVM rbf','SVM linear', 'Decision Tree', 'Random Forests', 'FFNN')
y_pos = np.arange(len(objects))

plt.barh(y_pos, hl_without_plot, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('Hamming Loss')
plt.title('20 newsgroups vectorized: Hamming Loss without standardization')


# In[52]:


hl_with_plot = [hl1_s, hl2_s, hl3_s_rbf, hl3_s_linear, hl4_s, hl5_s, hl6_s]
objects = ('Log reg', 'Perceptron', 'SVM rbf','SVM linear', 'Decision Tree', 'Random Forests', 'FFNN')
y_pos = np.arange(len(objects))

plt.barh(y_pos, hl_with_plot, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.ylabel('Hamming Loss')
plt.title('20 newsgroups vectorized: Hamming Loss with standardization')





