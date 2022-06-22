# Overview
Data needs to be loaded into the DLNN model for training and validating the model.



# Pytorch functions
## DataLoader
The [DataLoader](https://pytorch.org/docs/stable/data.html) function is used to load data and supports two types of datasets:
1. map-style datasets: such a dataset, when accessed with `dataset[idx]`, could read the `idx`-th image and its corresponding label from a folder on the disk
2. iterable-style datasets: such a dataset, when called `iter(dataset)`, could return a stream of data reading from a database, a remote server, or even logs generated in real time
The DataLoader function can load data in batches or one at a time. When `batch_size` (default `1`) is not `None`, the data loader yields batched samples instead of individual samples. `batch_size` and `drop_last` arguments are used to specify how the data loader obtains batches of dataset keys.

Dataloaders are iterables over the dataset. So when you iterate over it, it will return `B` randomly from the dataset collected samples (including the data-sample and the target/label), where `B` is the batch-size.

To create such a dataloader you will first need a class which inherits from the Dataset Pytorch class. There is a standard implementation of this class in pytorch which should be `TensorDataset`. But the standard way is to create your own. [Click this link to see the example](https://stackoverflow.com/questions/65138643/examples-or-explanations-of-pytorch-dataloaders)

# Code

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```

[Data loader in pytorch](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) 

Define a loading function:
```python
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
```