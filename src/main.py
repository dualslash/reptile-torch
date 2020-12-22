# Define Experiment Parameters
inner_step_size = 0.02
inner_batch_size = 5

outer_step_size = 0.1
outer_iterations = 1000
meta_batch_size = 10

eval_iterations = 32
eval_batch_size = 10
eval_range = range(1,11)

model_size = 32
sample_radius = 4
sample_count = 100

params = [inner_step_size, inner_batch_size,
          outer_step_size, outer_iterations, meta_batch_size,
          eval_iterations, eval_batch_size]

# Define Experiment Task and Model
task = logistic
log = SummaryWriter(data_folder)
model = Reptile(TorchModule(model_size), log, params)

# Train Model
eval_mse = np.empty(shape=[len(eval_range), eval_batch_size])
sample_space = meta_sample(sample_radius, sample_count)
model.train(task)

# Evaluate Model
for batch in range(eval_batch_size):

  samples, task_theta  = sample(task)

  for sample_size in eval_range:

    # Estimate Model for Batch
    estimate, loss = model.eval(samples, sample_size, eval_iterations, inner_step_size)
    eval_mse[sample_size-1, batch-1] = loss
    
    # Log Results to Tensorboard
    for idx in range(len(samples)):
        log.add_scalars('Model Evaluation {}/{} Samples'.format(batch + 1, sample_size), 
            {'Task': samples[idx], 
              'Baseline': estimate[0][idx][0], 
              'Estimate' : estimate[-1][idx][0]}, 
              idx)

log.close()
print(eval_mse.mean(axis=1)[:,None])