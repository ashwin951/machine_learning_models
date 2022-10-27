from distutils.log import Log
from typing_extensions import Self
import numpy as np

# create logistic regression class which contains necessary functions to run logisitc regression
class LogisticRegression:
	def sigmoid(self,z):
		"""
		Compute the sigmoid of z

		Args:
			z (ndarray): A scalar, numpy array of any size.

		Returns:
			 (ndarray): sigmoid(z), with the same shape as z
         

		"""
		return 1/(1 + np.exp(-z))

logReg = LogisticRegression()

