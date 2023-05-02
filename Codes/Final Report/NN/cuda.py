device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # For activating of CUDA
example_data = example_data.to(device) # To transfer data from CPU to GPU