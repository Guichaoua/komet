.. komet documentation master file, created by
   sphinx-quickstart on Sun Feb 18 12:55:08 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Komet - Kronecker Optimized Method for DTI Prediction
=====================================================

.. image:: ../../img/komet-logo-small.png
   :alt: Komet Logo
   :align: center

.. toctree::
   :hidden:
   :maxdepth: 1
   :glob:
   :caption: Getting started

   vignettes/*


.. toctree::
   :hidden: 
   :maxdepth: 3
   :caption: API

   komet


Overview
--------

This library is designed for computational biology and cheminformatics, focusing on the prediction and analysis of molecular interactions. It provides tools for loading and processing molecular and protein data, computing molecular fingerprints, estimating interaction probabilities, and evaluating model performance. This suite is particularly useful for researchers and developers working in drug discovery and molecular docking simulations.

Citation
--------

If you use this library, please be sure to cite::

   @article{YourLastName2024Molecular,
     title={Molecular Interaction Library: A Tool for Predicting and Analyzing Molecular Interactions},
     author={YourLastName, YourFirstName and CoAuthorLastName, CoAuthorFirstName},
     journal={Journal of Computational Biology and Cheminformatics},
     volume={XX},
     number={XX},
     pages={XX-XX},
     year={2024},
     publisher={PublisherName},
     doi={10.1234/jcbci.2024.56789},
     url={http://www.example.com/library-article},
     abstract={This article introduces the Molecular Interaction Library, a comprehensive Python toolkit designed for the prediction and analysis of molecular interactions in computational biology and cheminformatics. Covering functionalities from data loading and preprocessing to molecular feature computation, model training, prediction, and evaluation, this library aims to facilitate research and development in drug discovery and molecular docking simulations. We detail the implementation of key features, such as Morgan fingerprint computation, Nystrom approximation for kernel methods, and SVM-based predictive modeling with L-BFGS optimization, alongside a case study demonstrating its application in identifying potential drug candidates.}
   }

Dependencies
------------

The library requires the following Python packages:

- pandas
- numpy
- rdkit
- torch
- scikit-learn
- zipfile
- pickle

Installation
------------

To install the required dependencies, run::

   pip install pandas numpy rdkit-pypi torch scikit-learn

Functionalities
---------------

Data Loading and Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``load_df(name)``: Loads a dataframe from a CSV file, cleans up SMILES strings that cannot be read by RDKit.
- ``add_indsmiles(df)``: Adds indices for each unique SMILES string in the dataframe.
- ``add_indfasta(df)``: Adds an index column for each unique FASTA sequence in the dataframe.

Molecular Features Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``Morgan_FP(list_smiles)``: Computes the Morgan fingerprints for a list of SMILES strings.
- ``Nystrom_X(smiles_list, S, MorganFP, V, rM, Mu, epsi)``: Computes the approximate features of the molecular kernel using the Nystrom approximation.

Model Training and Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``SVM_bfgs(X_cn, Y_cn, y, I, J, lamb)``: Trains an SVM model using the L-BFGS optimization algorithm.
- ``compute_proba_Platt_Scalling(w_bfgs, X_cn, Y_cn, y, I, J)``: Computes probability estimates for interaction predictions using Platt scaling.
- ``compute_proba(w_bfgs, b_bfgs, s, t, X_cn, Y_cn, I, J)``: Computes probabilities using the trained weights and Platt scaling parameters.

Evaluation
~~~~~~~~~~

- ``results(y, y_pred, proba_pred)``: Computes and returns various performance metrics of the model.

Example Usage
-------------

.. code-block:: python

   import pandas as pd

   # Load your dataset
   df = load_df("molecule_data.csv")

Contributing
------------

We welcome contributions to this library. If you have suggestions for improvements or bug fixes, please open an issue or a pull request.

License
-------

`MIT License <_static/LICENSE>`_ 

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`