import click
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
import urllib.request

@click.command()
@click.option(
              '--dataset',
              help='Identifier for the dataset we want to use',
              default='make_moons'
              )
@click.option(
              '--outfile',
              help='Path to file which will contain pickled ML input',
              default='ml_output.pkl'
              )

# Function to perform an ml classification on an input dataset using various classifying algorithms
def run_classifier(dataset, outfile):
  h = .02  # step size in the mesh

  names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
           "Decision Tree"]

  classifiers = [
      KNeighborsClassifier(3),
      SVC(kernel="linear", C=0.025),
      SVC(gamma=2, C=1),
      GaussianProcessClassifier(1.0 * RBF(1.0)),
      DecisionTreeClassifier(max_depth=5),
      ]

  # Get the data from the download link
  this_dataset = np.genfromtxt(urllib.request.urlopen(url=dataset), skip_header=1)
  X = this_dataset[:,0:2]
  y = this_dataset[:,2]

  figure = plt.figure(figsize=(15, 3))
  i = 1
  X = StandardScaler().fit_transform(X)
  X_train, X_test, y_train, y_test = \
      train_test_split(X, y, test_size=.4, random_state=42)

  x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
  y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                       np.arange(y_min, y_max, h))

  # just plot the dataset first
  cm = plt.cm.RdBu
  cm_bright = ListedColormap(['#FF0000', '#0000FF'])
  ax = plt.subplot(1, len(classifiers) + 1, i)
  ax.set_title("Input data")
  # Plot the training points
  ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
             edgecolors='k')
  # Plot the testing points
  ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
             edgecolors='k')
  ax.set_xlim(xx.min(), xx.max())
  ax.set_ylim(yy.min(), yy.max())
  ax.set_xticks(())
  ax.set_yticks(())
  i += 1

  # iterate over classifiers
  for name, clf in zip(names, classifiers):
      ax = plt.subplot(1, len(classifiers) + 1, i)
      clf.fit(X_train, y_train)
      score = clf.score(X_test, y_test)

      # Plot the decision boundary. For that, we will assign a color to each
      # point in the mesh [x_min, x_max]x[y_min, y_max].
      if hasattr(clf, "decision_function"):
          Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
      else:
          Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

      # Put the result into a color plot
      Z = Z.reshape(xx.shape)
      ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

      # Plot the training points
      ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                 edgecolors='k')
      # Plot the testing points
      ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                 edgecolors='k', alpha=0.6)

      ax.set_xlim(xx.min(), xx.max())
      ax.set_ylim(yy.min(), yy.max())
      ax.set_xticks(())
      ax.set_yticks(())
      ax.set_title(name)
      ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
              size=15, horizontalalignment='right')
      i += 1

  plt.tight_layout()
  plt.savefig(outfile)

if __name__ == "__main__":
  run_classifier()


