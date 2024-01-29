from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def compute_mae(true_values, predicted_values):
    return mean_absolute_error(true_values, predicted_values)


def compute_rmse(true_values, predicted_values):
    return np.sqrt(mean_squared_error(true_values, predicted_values))


def compute_wmape(true_values, predicted_values):
    return np.sum(np.abs(true_values - predicted_values)) / np.sum(true_values)


def compute_r2(true_values, predicted_values):
    return r2_score(true_values, predicted_values)