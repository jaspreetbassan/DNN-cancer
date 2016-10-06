from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# Data sets
CANCER_TRAINING = "cancer_train.csv"
CANCER_TEST = "cancer_test.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv(filename=CANCER_TRAINING,
                                                       target_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv(filename=CANCER_TEST,
                                                   target_dtype=np.int)

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=30)]

# Build 4 layer DNN with 10, 20, 20, 20 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 20, 20],
                                            n_classes=2,
                                            model_dir="/tmp/cancer1_model")

# Fit model.
classifier.fit(x=training_set.data,
               y=training_set.target,
steps=2000)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

# Classify two new flower samples.
#new_samples = np.array(
 #   [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
#y = classifier.predict(new_samples)
#print('Predictions: {}'.format(str(y)))



