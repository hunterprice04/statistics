from pyspark import RDD
from pyspark.mllib.stat import Statistics
from math import sqrt

def mean(rdd: RDD) -> float:
	return rdd.sum() / float(rdd.count())


def median(rdd: RDD) -> float:
	sorted_and_indexed = rdd.sortBy(lambda x: x).zipWithIndex().map(lambda v, k: (k, v))
	n = sorted_and_indexed.count()
	if n % 2 == 1:
		return sorted_and_indexed.lookup((n-1) / 2)[0]
	else:
		v1 = sorted_and_indexed.lookup(n / 2)[0]
		v2 = sorted_and_indexed.lookup((n / 2) - 1)[0]
		return (v1 + v2) / 2.0


def stdev(rdd: RDD, mean: float) -> float:
	return sqrt(rdd.map(lambda x: pow(x-mean, 2)).sum() / rdd.count())


def skewness(rdd: RDD, mean: float, stdev: float) -> float:
	return rdd.map(lambda x: pow(x-mean, 3)).sum() / (pow(stdev, 3)*rdd.count())


def kurtosis(rdd: RDD, mean: float, stdev: float) -> float:
	return rdd.map(lambda x: pow(x-mean, 4)).sum() / (pow(stdev, 4)*rdd.count())


def covariance(rdd1: RDD, mean1: float, rdd2: RDD, mean2: float) -> float:
	rdd_zipped = rdd1.zip(rdd2)
	return rdd_zipped.map(lambda x, y: (x-mean1)*(y-mean2)).sum() / rdd_zipped.count()


def correlation(cov: float, stdev1: float, stdev2: float) -> float:
	return cov / (stdev1 * stdev2)


def correlation_matrix(arg1: RDD, *args: RDD):
	data = arg1
	for rdd in args:
		data.zip(rdd)
	data = data.map(lambda a, b, c, d: [a, b, c, d])
	return Statistics.corr(data)
