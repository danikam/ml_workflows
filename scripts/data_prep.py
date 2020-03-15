import click
import numpy as np
import pickle
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

# Function to download and read in the specified dataset, and pre-process the data for ML fitting
def preprocess_data(dataset, outfile):
  
  # Get the data from the download link
  this_dataset = np.genfromtxt(urllib.request.urlopen(url=dataset), skip_header=1)
  X = this_dataset[:,0:2]
  y = this_dataset[:,2]

  # preprocess dataset, split into training and test part
  X = StandardScaler().fit_transform(X)
  X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)

  # Make a dictionary of the preprocessed ML input data
  ml_input ={
  'X': X,
  'X_train': X_train,
  'X_test': X_test,
  'y_train': y_train,
  'y_test': y_test,
  }

  # Save the preprocessed ML input as a pickle file
  with open(outfile, 'wb') as f_out:
    pickle.dump(ml_input, file=f_out)

if __name__ == "__main__":
  preprocess_data()
