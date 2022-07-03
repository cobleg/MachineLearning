# Overview
[Python](https://www.python.org/) is a computer programming language required to run Pytorch. Within the Python ecosystem there are both libraries and frameworks. See this article to understand the distinction: [Python Frameworks vs. Python Libraries](https://fullscale.io/blog/python-frameworks-vs-python-libraries/). 

# Relevant libraries
Relevant Python libraries are:

`os`: The Python [os library](https://www.linuxscrew.com/python-os) contains tools and functions for interacting with the underlying Operating System being used to execute the Python code.
`tarfile`: provides read-write capability for tar archive files.
`zipfile`: ability to work with zip compressed files
`requests`: provides ability to request a web page or file
`numpy`: NumPy contains scientific computation functions such as statistical computing. 
`pandas`:  helps with manipulating, processing, cleaning, and crunching datasets in Python.
 `scikit-learn`: is an open source machine learning library that supports supervised and unsupervised learning, see [Getting started](https://scikit-learn.org/stable/getting_started.html) for details. 
# Relevant frameworks
Relevant [python frameworks](https://www.simplilearn.com/python-frameworks-article) are:
- [[PyTorch]]

# Helper functions
This section contains custom function recipes that enable various parts of the DLNN process. Helper functions help minimise the manual input required to create a model.

## Download function
This function provides the ability to download files.

```python
def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """Download a file inserted into DATA_HUB, return the local filename."""
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576) # defines the size of the file being downloaded.
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Hit cache
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname
```

### Notes
`hashlib.sha1()` is a hashing function. **Hashing algorithms** are mathematical functions that convert data into fixed-length hash values, hash codes, or hashes. The output hash value is literally a summary of the original value. The most important thing about these hash values is that **it is impossible to retrieve the original input data just from hash values**. Hashing values provides a way of managing data integrity; you'll be able to tell if some file isn't modified since creation ([data integrity](https://en.wikipedia.org/wiki/Data_integrity "Data Integrity"))[How to Use Hashing Algorithms in Python using hashlib](https://www.thepythoncode.com/article/hashing-functions-in-python-using-hashlib) 

## k-fold cross-validation
See this note for details on [[cross-validation]]. 
# Notes
`//`: The double slash (//) operator is used in python for different purposes. One use of this operator is to get the division result. The division result of two numbers can be an integer or a floating-point number. In python version 3+, both the single slash (/) operator and the double slash (//) operator are used to get the division result containing the floating-point value. One difference is that the single slash operator returns proper output for the floating-point result, but the double slash operator can’t return the fractional part of the floating-point result. [Use of Python double slash (//)](https://linuxhint.com/use-python-double-slash/)

`skill`: In the fields of [forecasting](https://en.wikipedia.org/wiki/Forecasting "Forecasting") and [prediction](https://en.wikipedia.org/wiki/Prediction "Prediction"), **forecast skill** or **prediction skill** is any measure of the accuracy and/or degree of association of prediction to an observation or estimate of the actual value of what is being predicted (formally, the predictand); it may be quantified as a **skill score**. In [meteorology](https://en.wikipedia.org/wiki/Meteorology "Meteorology"), more specifically in [weather forecasting](https://en.wikipedia.org/wiki/Weather_forecasting "Weather forecasting"), skill measures the superiority of a forecast over a simple historical baseline of past observations.

`torch.cat`: Concatenates the given sequence of `seq` tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty. [TORCH.CAT](https://pytorch.org/docs/stable/generated/torch.cat.html)

