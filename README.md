

## Installation
### Dependencies
The hiDF package requires 

- Python (>= 3.3)
- Numpy (>= 1.8.2)
- Scipy (>= 0.13.3)
- Cython
- pydotplus
- matplotlib 
- jupyter 
- pyyaml
- scikit-learn (>= 0.22)

Before the installation, please make sure you installed the above python packages correctly via pip:
```bash
pip install cython numpy scikit-learn pydotplus jupyter pyyaml matplotlib
```
### Basic setup and installation

Installing hiDF package is simple. Just clone this repo and use pip install.


Go to the `hiDF` folder and use pip install:
```bash
pip install -e .
```
If hiDF is installed successfully, you should be able to see it using pip list:
```bash
pip list | grep hiDF
```
and you should be able to run all the tests (assume the working directory is in the package hiDF):
```bash
python hiDF/hiDFtest/test.py
```

#### Result of Adult Dataset
```
Accuracy: 86.90 ± 0.05  
F1:    86.43 ± 0.07
```
