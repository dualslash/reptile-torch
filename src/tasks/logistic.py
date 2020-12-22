def logistic(x, theta):

  return theta[0] / (1 + np.exp(-1 * theta[1] * ( x - theta[2])))