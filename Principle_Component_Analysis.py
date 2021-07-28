import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('D:/Python Workout/wine.data', header=None)
# separate train and test data
from sklearn.model_selection import train_test_split
X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# step: 1 - standardize the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

# step:2 construct the covariant matrix
import numpy as np
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)
print('\nEigenvalues \n%s', eigen_vals)

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
        label='Individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid',
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('images/05_02.png', dpi=300)
plt.show()

# step - 3: sort eigenvalues by decreasing order to rank eigenvectors
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
# (eigenvalue, eigenvector) tuple
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

# collect two eigenvectors based on two largest eigenvalues
# w = projection matrix which transforms data into lower-dimensional subspace
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],  # 13 by 2 dimensional
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)

# transform x onto new PCA subspace
X_train_std[0].dot(w)
# transform entire dataset
X_train_pca = X_train_std.dot(w)

# visualize transformed wine dataset
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0],
                X_train_pca[y_train == l, 1],
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_03.png', dpi=300)
plt.show()


