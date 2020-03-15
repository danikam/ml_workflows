import click
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

@click.command()
@click.option(
              '--prepped_data',
              help='Path to prepared data',
              )
@click.option(
              '--classifier',
              help='name of classifier to use',
              default='Nearest Neighbors'
              )

@click.option(
              '--outfile',
              help='Path to save classification result to',
              default='ml_output.pkl'
              )

# Function to classify the input prepped data using the specified ML classifier
def run_classifier(prepped_data, classifier, outfile):
  
  classifier_dict = {
  'nearest_neighbours': KNeighborsClassifier(3),
  'linear_svm': SVC(kernel="linear", C=0.025),
  'rbf_svm': SVC(gamma=2, C=1),
  'gaussian_process': GaussianProcessClassifier(1.0 * RBF(1.0)),
  'decision_tree': DecisionTreeClassifier(max_depth=5),
  }
  
  # Specify the classifier
  try: clf = classifier_dict[classifier]
  except KeyError: clf = classifier_dict['Nearest Neighbors']

  # Read in the prepped data
  with open(prepped_data, 'rb') as f:
    ml_input = pickle.load(f)

  # Fit with the specified classifier
  clf.fit(ml_input['X_train'], ml_input['y_train'])
  score = clf.score(ml_input['X_test'], ml_input['y_test'])

  # Dictionary to contain the ML fit result, and related info
  ml_output = {
  'name': classifier,
  'clf': clf,
  'score': score
  }

  # Save the classifier result
  with open(outfile, 'wb') as f_out:
      pickle.dump(ml_output, file=f_out)

if __name__ == "__main__":
  run_classifier()
