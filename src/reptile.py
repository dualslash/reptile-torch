class Reptile:

  def __init__(self, model, log, params):

    # Intialize Reptile Parameters
    self.inner_step_size = params[0]
    self.inner_batch_size = params[1]
    self.outer_step_size = params[2]
    self.outer_iterations = params[3]
    self.meta_batch_size = params[4] 
    self.eval_iterations = params[5] 
    self.eval_batch_size = params[6]

    # Initialize Torch Model and Tensorboard
    self.model = model.to(device)
    self.log = log

  def reset(self):

    # Reset Training Gradients
    self.model.zero_grad()
    self.current_loss = 0
    self.current_batch = 0

  def train(self, task):

    # Train from Scratch
    self.reset()

    # Outer Training Loop
    for outer_iteration in tqdm.tqdm(range(self.outer_iterations)):

      # Track Current Weights
      current_weights = deepcopy(self.model.state_dict())

      # Sample a new Subtask
      samples, task_theta = sample(task)

      # Inner Training Loop
      for inner_iteration in range(self.inner_batch_size):

        # Process Meta Learning Batches
        for batch in range(0, len(sample_space), self.meta_batch_size):

          # Get Permuted Batch from Sample
          perm = np.random.permutation(len(sample_space))
          idx = perm[batch: batch + self.meta_batch_size][:, None]

          # Calculate Batch Loss
          batch_loss = self.loss(sample_space[idx], samples[idx])
          batch_loss.backward()

          # Update Model Parameters
          for theta in self.model.parameters():

            # Get Parameter Gradient
            grad = theta.grad.data

            # Update Model Parameter
            theta.data -= self.inner_step_size * grad

          # Update Model Loss from Torch Model Tensor
          loss_tensor = batch_loss.cpu()
          self.current_loss += loss_tensor.data.numpy()
          self.current_batch += 1

      # Linear Cooling Schedule
      alpha = self.outer_step_size * (1 - outer_iteration / self.outer_iterations)

      # Get Current Candidate Weights
      candidate_weights = self.model.state_dict()

      # Transfer Candidate Weights to Model State Checkpoint
      state_dict = {candidate: (current_weights[candidate] + alpha * 
                               (candidate_weights[candidate] - current_weights[candidate])) 
                                for candidate in candidate_weights}
      self.model.load_state_dict(state_dict)
      
      # Log new Training Loss
      self.log.add_scalars('Model Estimate/Loss', 
                           {'Loss' : self.current_loss / self.current_batch}, 
                           outer_iteration)

  def loss(self, x, y):

    # Reset Torch Gradient
    self.model.zero_grad()

    # Calculate Torch Tensors
    x = torch.tensor(x, device = device, dtype = torch.float32)
    y = torch.tensor(y, device = device, dtype = torch.float32)

    # Estimate over Sample
    yhat = self.model(x)

    # Regression Loss over Estimate
    loss = nn.MSELoss()
    output = loss(yhat, y)

    return output

  def predict(self, x):

    # Estimate using Torch Model
    t = torch.tensor(x, device = device, dtype = torch.float32)
    t = self.model(t)

    # Bring Torch Tensor from GPU to System Host Memory
    t = t.cpu()

    # Return Estimate as Numpy Float
    y = t.data.numpy()

    return y

  def eval(self, base_truth, meta_batch_size, gradient_steps, inner_step_size):

    # Sample Points from Task Sample Space
    x, y = sample_points(base_truth, self.meta_batch_size)

    # Model Base Estimate over Sample Space
    estimate = [self.predict(sample_space[:,None])]

    # Store Meta-Initialization Weights
    meta_weights = deepcopy(self.model.state_dict())

    # Get Estimate Loss over Meta-Initialization
    loss_t = self.loss(x,y).cpu()
    meta_loss = loss_t.data.numpy()

    # Calculcate Estimate over Gradient Steps
    for step in range(gradient_steps):

      # Calculate Evaluation Loss and Backpropagate
      eval_loss = self.loss(x,y)
      eval_loss.backward()

      # Update Model Estimate Parameters
      for theta in self.model.parameters():

        # Get Parameter Gradient
        grad = theta.grad.data

        # Update Model Parameter
        theta.data -= self.inner_step_size * grad

      # Update Estimate over Sample Space
      estimate.append(self.predict(sample_space[:, None]))

    # Get Estimate Loss over Evaluation
    loss_t = self.loss(x,y).cpu()
    estimate_loss = loss_t.data.numpy()
    evaluation_loss = abs(meta_loss - estimate_loss)/meta_batch_size
    
    # Restore Meta-Initialization Weights
    self.model.load_state_dict(meta_weights)

    return estimate, evaluation_loss