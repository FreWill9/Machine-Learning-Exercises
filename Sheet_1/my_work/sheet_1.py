from sklearn.datasets import load_digits
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import colors

# Exercise 1
digits = load_digits()
print(digits.keys())
data = digits["data"]
images = digits["images"]
target = digits["target"]
target_names = digits["target_names"]

img = np.array(images[19])

print(img.shape)

assert 2 == len(img.shape)

# plt.figure()
# plt.gray()

# plt.imshow(img, interpolation="nearest")
# plt.show()

# plt.imshow(img, interpolation="bicubic")
# plt.show()

mask = [x for x in range(0, len(target)-1) if ((target[x] == 3) or (target[x] == 9))]

X_all = [data[x] for x in mask]
y_all = [target[x] for x in mask]

X_train, X_test, y_train, y_test =\
    model_selection.train_test_split(X_all, y_all,
                                     test_size=0.4, random_state=0)

# Exercise 2
# 2.1


def features2d(data_matrix):
    feature_space = []
    considered_pixel_locations = [17, 18, 19, 25, 26]
    considered_pixel_locations_sec = [22, 23, 29, 30, 31, 38, 39]
    for instance in data_matrix:
        # feature 1: sum over considered pixels, performed best with sum < 20
        considered_pixels = [instance[x] for x in considered_pixel_locations]
        sum_over_considered_pixels = sum(considered_pixels)

        # feature 2: sum over secondary array of considered pixels
        considered_pixels_sec = [instance[x] for x in considered_pixel_locations_sec]
        sum_over_considered_pixels_sec = sum(considered_pixels_sec)

        feature_space.append([sum_over_considered_pixels, sum_over_considered_pixels_sec])
    return feature_space


x = X_all     # list of numpy-arrays of one-dimensional image data
features = features2d(x)

# 2.2
features_threes = [features[x] for x in range(0, len(y_all)) if (y_all[x] == 3)]
features_nines = [features[x] for x in range(0, len(y_all)) if (y_all[x] == 9)]
x_axis_threes = [x[1] for x in features_threes]
y_axis_threes = [x[0] for x in features_threes]
x_axis_nines = [x[1] for x in features_nines]
y_axis_nines = [x[0] for x in features_nines]

plt.ylabel("Total brightness in up-left region")
plt.xlabel("Total brightness in middle-right region")

plt.scatter(x_axis_threes, y_axis_threes, s=20, c='green', marker='o')
plt.scatter(x_axis_nines, y_axis_nines, s=20, c='red', marker='x')
plt.show()

# 2.3


def nearest_mean(training_feat, training_lab, test_feat):
    x_values_threes = []
    y_values_threes = []
    x_values_nines = []
    y_values_nines = []
    count = 0
    for instance in training_feat:
        if training_lab[count] == 3:
            x_values_threes.append(instance[1])
            y_values_threes.append(instance[0])
        else:
            x_values_nines.append(instance[1])
            y_values_nines.append(instance[0])
        count += 1
    mean_threes = [sum(x_values_threes)/len(x_values_threes), sum(y_values_threes)/len(y_values_threes)]
    mean_nines = [sum(x_values_nines)/len(x_values_nines), sum(y_values_nines)/len(y_values_nines)]

    return [-1 if (math.dist(x, mean_threes) < math.dist(x, mean_nines)) else 1 for x in test_feat]


training_features = features2d(X_train)
training_labels = y_train
test_features = features2d(X_test)
predicted_labels = nearest_mean(training_features, training_labels, test_features)

# test classifier
predicted_labels_training = nearest_mean(training_features, training_labels, training_features)
miss_counter_training = [1 for x in range(0, len(predicted_labels_training)-1) if (predicted_labels_training[x] == -1 and y_train[x] == 9) or (predicted_labels_training[x] == 1 and y_train[x] == 3)]
miss_rate_training = len(miss_counter_training) / len(predicted_labels_training)
print(f"Training error rate: {miss_rate_training}")

miss_counter_test = [1 for x in range(0, len(predicted_labels)-1) if (predicted_labels[x] == -1 and y_test[x] == 9) or (predicted_labels[x] == 1 and y_test[x] == 3)]
miss_rate_test = len(miss_counter_test) / len(predicted_labels)
print(f"Test error rate: {miss_rate_test}")

# 2.4
grid_data = []
for i in range(0, 50):
    grid_data.append([])
    for j in range(0, 50):
        grid_data[i].append(nearest_mean(training_features, training_labels, [[i, j]]))

# create discrete colormap
cmap = colors.ListedColormap(['green', 'red'])
bounds = [-2, 0, 2]
norm = colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots()
ax.imshow(grid_data, cmap=cmap, norm=norm, alpha=0.5, origin='lower')

plt.show()
