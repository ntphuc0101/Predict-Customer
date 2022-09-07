# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
In this project, we will identify credit card customers that are most likely to churn. The Project will include a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package will also have the flexibility of being run interactively or from the command-line interface (CLI).

## Files and data description
Overview of the files and data present in the root directory.

```
.
├── churn_library.py
├── churn_notebook.ipynb
├── churn_script_logging_and_tests.py
├── data
│   └── bank_data.csv
├── Guide.ipynb
├── images
│   ├── eda
│   │   ├── Churn.png
│   │   ├── Customer_Age.png
│   │   ├── Gender.png
│   │   ├── heatmap.png
│   │   ├── Marital_Status.png
│   │   └── Total_Trans_Amt.png
│   └── results
│       ├── feature_importances.png
│       ├── logistic_results.png
│       ├── rf_results.png
│       └── roc_curve_result.png
├── logs
│   └── churn_library.log
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── __pycache__
│   ├── churn_library2.cpython-36.pyc
├── README.md
└── requirements.txt
```



These dataset is downloaded from Kaggle, and its name is Credit Card customers.
1. `churn_library.py` - churn customer analysis
2. `churn_script_logging_and_testing.py` - testing functions from churn_library.py
3. `data folder` - Credit Card customers
4. `images` - output images 
5. `logs` - logging files
6. `models` - model files

## Running Files

There are several steps to run this projects -  #git and #conda:
#git - for git clone link to cloning projects
#conda - for activating environment

1. Clone repository from github

Git clone

2. Create a conda environment

python -m pip install -r requirements_py3.8.txt

3. Activate the conda environment

     conda activate env


For Installing the linter and auto-formatter: pip install pylint pip install autopep8

Run: python churn_library.py for data analysis
     python python_script_logging_and_tests.py for testing functions of churn prediction

