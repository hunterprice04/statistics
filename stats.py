import statistics
import numpy as np

import scipy.stats as stats
from math import sqrt


def mean(a) -> float:
	return np.average(a)


def median(a):
	return np.median(a)


def stdev(a):
	return np.std(a)


def skewness(a):
	return stats.skew(a)


def kurtosis(a):
	return stats.kurtosis(a)


def covariance(a1, mean1: float, a2, mean2: float):
	return sum(map(lambda x, y: (x-mean1)*(y-mean2), a1, a2)) / len(a1)

def correlation(cov, stdev1, stdev2):
	return cov / (stdev1 * stdev2)


def correlation_matrix(*args: list):
	return np.corrcoef(args)

if __name__ == '__main__':
	x = [1,2,3,4,5,6,7,8,9,10]
	y = [7,6,5,4,5,6,7,8,9,10]

	print(covariance(x, mean(x), y, mean(y)))
	print(correlation(covariance(x, mean(x), y, mean(y)), stdev(x), stdev(y)))

	print(correlation_matrix(x, y))