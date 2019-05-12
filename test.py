from __future__ import print_function

import math

import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 9
pd.options.display.float_format = '{:.1f}'.format

graduate_admission_df = pd.read_csv("/home/supimi/ML_project/graduate-admissions/Admission_Predict_Ver1.1.csv", sep=",")

# Re-index the data records
graduate_admission_df = graduate_admission_df.reindex(np.random.permutation(graduate_admission_df.index))

prob_of_admission = graduate_admission_df[["Chance_of_Admit"]]
prob_of_admission = np.array(list(dict(prob_of_admission).values())[0])

gre_score = graduate_admission_df[["GRE_Score"]]
gre_score = np.array(list(dict(gre_score).values())[0])

print(gre_score)


plt.xlabel("GRE Score")
plt.ylabel("Chance of Admit")
plt.title("GRE Score Vs. Chane of Admit")
plt.scatter(gre_score, prob_of_admission)
plt.show()


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(buffer_size=100)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_model(learning_rate, steps, batch_size, periods):
    steps_per_period = steps / periods
    targets = graduate_admission_df["Chance_of_Admit"]

    # Define the input feature: total_rooms.
    my_features = graduate_admission_df[["GRE_Score", "CGPA", "TOEFL_Score", "University_Rating", "LOR"]]

    # Configure a numeric feature column for total_rooms.
    feature_columns = [tf.feature_column.numeric_column("GRE_Score"), tf.feature_column.numeric_column("CGPA"),
                       tf.feature_column.numeric_column("TOEFL_Score"),
                       tf.feature_column.numeric_column("University_Rating"),
                       tf.feature_column.numeric_column("LOR")]

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=optimizer
    )

    rmse_list = []

    training_fn = lambda: my_input_fn(features=my_features, targets=targets, batch_size=batch_size, num_epochs=2)
    predict_fn = lambda: my_input_fn(features=my_features, targets=targets, num_epochs=1, shuffle=False)

    for p in range(periods):
        linear_regressor.train(
            training_fn,
            steps=steps_per_period

        )

        predictions = linear_regressor.predict(input_fn=predict_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])

        rmse = math.sqrt(metrics.mean_squared_error(predictions, targets))
        print("RMSE - ", p, " ", rmse)
        rmse_list.append(rmse)

    print("Model training ended.")

    rmse_list = np.array(rmse_list)

    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.plot(rmse_list)
    plt.show()


train_model(
    learning_rate=0.00001,
    steps=100,
    batch_size=5,
    periods=10
)

# print(graduate_admission_df.describe())


# print(graduate_admission_df.head())
