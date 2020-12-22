class TorchModule(nn.Module):

  def __init__(self, n):

    # Initialize PyTorch Base Module
    super(TorchModule, self).__init__()

    # Define Multi-Layer Perceptron
    self.input = nn.Linear(1,n)
    self.hidden_in = nn.Linear(n,n)
    self.hidden_out = nn.Linear(n,n)
    self.output = nn.Linear(n,1)

  def forward(self, x):

    # PyTorch Feed Forward Subroutine
    x = torch.tanh(self.input(x))
    x = torch.tanh(self.hidden_in(x))
    x = torch.tanh(self.hidden_out(x))
    y = self.output(x)

    return y