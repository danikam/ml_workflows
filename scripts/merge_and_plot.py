import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

h = .02  # step size in the mesh

# Read in the user arguments
ml_input_file = sys.argv[1]
outfile = sys.argv[2]
ml_files = sys.argv[3:]

# Read in the info for the input dataset (needed for plotting)
with open(ml_input_file, 'rb') as f:
  ml_input = pickle.load(f)

i = 1

# Make a figure to contain all plots
figure = plt.figure(figsize=(15, 3))

# Prepare the x-y grid for plotting
x_min, x_max = ml_input['X'][:, 0].min() - .5, ml_input['X'][:, 0].max() + .5
y_min, y_max = ml_input['X'][:, 1].min() - .5, ml_input['X'][:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Prepare the colour map for plotting points and fit results
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

# Plot the input dataset
ax = plt.subplot(1, len(ml_files) + 1, i)
ax.set_title("Input data")
ax.scatter(ml_input['X_train'][:, 0], ml_input['X_train'][:, 1], c=ml_input['y_train'], cmap=cm_bright, edgecolors='k')   # Training points
ax.scatter(ml_input['X_test'][:, 0], ml_input['X_test'][:, 1], c=ml_input['y_test'], cmap=cm_bright, alpha=0.6, edgecolors='k')   # Testing points
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
i += 1

# Loop through pickle files containing ML fit results for each classifier, and plot the classification result
for ml_file in ml_files:
  print(ml_file)
  with open(ml_file, 'rb') as f: ml_result = pickle.load(f)
  
  ax = plt.subplot(1, len(ml_files) + 1, i)

  # Plot the decision boundary. For that, we will assign a color to each
  # point in the mesh [x_min, x_max]x[y_min, y_max].
  if hasattr(ml_result['clf'], "decision_function"):
    Z = ml_result['clf'].decision_function(np.c_[xx.ravel(), yy.ravel()])
  else:
    Z = ml_result['clf'].predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

  # Put the result into a color plot
  Z = Z.reshape(xx.shape)
  ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

  # Plot the training points
  ax.scatter(ml_input['X_train'][:, 0], ml_input['X_train'][:, 1], c=ml_input['y_train'], cmap=cm_bright, edgecolors='k')
  # Plot the testing points
  ax.scatter(ml_input['X_test'][:, 0], ml_input['X_test'][:, 1], c=ml_input['y_test'], cmap=cm_bright, edgecolors='k', alpha=0.6)

  ax.set_xlim(xx.min(), xx.max())
  ax.set_ylim(yy.min(), yy.max())
  ax.set_xticks(())
  ax.set_yticks(())
  ax.set_title(ml_result['name'])
  ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % ml_result['score']).lstrip('0'),
      size=15, horizontalalignment='right')

  i += 1

plt.tight_layout()
plt.savefig(sys.argv[2])

