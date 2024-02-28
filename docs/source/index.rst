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
     title={Advancing Drug-Target Interactions Prediction: Leveraging a Large-Scale Dataset with a Rapid and Robust Chemogenomic Algorithm},
     author={View ORCID ProfileGwenn Guichaoua, Philippe Pinel, Brice Hoffmann, Chloé-Agathe Azencott, Véronique Stoven},
     journal={BioRxiv},
      year={2024},
     doi={10.1101/2024.02.22.581599},
     url={https://www.biorxiv.org/content/10.1101/2024.02.22.581599v1},
     abstract={Predicting drug-target interactions (DTIs) is crucial for drug discovery, and heavily relies on supervised learning techniques. In the context of DTI prediction, supervised learning algorithms use known DTIs to learn associations between molecule and protein features, allowing for the prediction of new interactions based on learned patterns. In this paper, we present a novel approach addressing two key challenges in DTI prediction: the availability of large, high-quality training datasets and the scalability of prediction methods. First, we introduce LCIdb, a curated, large-sized dataset of DTIs, offering extensive coverage of both the molecule and druggable protein spaces. Notably, LCIdbcontains a much higher number of molecules, expanding coverage of the molecule space compared to traditional benchmarks. Second, we propose Komet (Kronecker Optimized METhod), a DTI prediction pipeline designed for scalability without compromising performance. Komet leverages a three-step framework, incorporating efficient computation choices tailored for large datasets and involving the Nyström approximation. Specifically, Komet employs a Kronecker interaction module for (molecule, protein) pairs, which is sufficiently expressive and whose structure allows for reduced computational complexity. Our method is implemented in open-source software, lever-aging GPU parallel computation for efficiency. We demonstrate the efficiency of our approach on various datasets, showing that Komet displays superior scalability and prediction performance compared to state-of-the-art deep-learning approaches. Additionally, we illustrate the generalization properties of Komet by showing its ability to solve challenging scaffold-hopping problems gathered in the publicly available LH benchmark. Komet is available open source at https://komet.readthedocs.io.}

Dependencies
------------

The library requires the following Python packages:

- pandas
- numpy
- rdkit
- torch
- scikit-learn
- zipp
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