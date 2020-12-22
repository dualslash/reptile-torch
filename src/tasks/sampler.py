def sample(task):

  if task is not logistic:

    raise NotImplementedError

  # Parametric Generator for Logistic Regression Task (TODO: Generalize for Task - Parameter Specification)
  theta = [np.random.uniform( 1, 10), 
           np.random.uniform( 1, 10),
           np.random.uniform(-1,  1)]

  return task(sample_space, theta), theta

def sample_points(task, batch_size):

  # Sample Random Points from Sample Space
  idx = np.random.choice(np.arange(len(sample_space)), batch_size, replace = False)
  return sample_space[idx[:,None]], task[idx[:,None]]

def meta_sample(radius, count):

  # Generate Sample Space of Specified Radius
  sample_space = np.linspace(-radius, radius, count)
  return sample_space