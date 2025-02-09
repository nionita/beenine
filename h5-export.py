import h5py
with h5py.File('model.h5', 'w') as f:
    for name, param in model.named_parameters():
        f.create_dataset(name, data=param.detach().numpy())

