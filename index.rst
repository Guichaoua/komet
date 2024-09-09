.. Komet - Kronecker Optimized Method for DTI Prediction documentation master file

Komet - Kronecker Optimized Method for DTI Prediction
=====================================================

.. image:: images/komet-logo-small.png
   :alt: Komet Logo
   :align: center

Overview
--------

This library is designed for computational biology and cheminformatics, focusing on the prediction and analysis of molecular interactions. It provides tools for loading and processing molecular and protein data, computing molecular fingerprints, estimating interaction probabilities, and evaluating model performance. This suite is particularly useful for researchers and developers working in drug discovery and molecular docking simulations.

Citation
--------

If you use this library, please be sure to cite::

@article{Komet2024,
   title={Drug–Target Interactions Prediction at Scale: The Komet Algorithm with the LCIdb Dataset},
   author={Gwenn Guichaoua, Philippe Pinel, Brice Hoffmann, Chloé-Agathe Azencott, Véronique Stoven},
   journal={Journal of Chemical Information and Modeling},
   year={2024},
   doi={10.1021/acs.jcim.4c00422},
   url={https://pubs.acs.org/doi/full/10.1021/acs.jcim.4c00422},
   abstract={Drug–target interactions (DTIs) prediction algorithms are used at various stages of the drug discovery process. In this context, specific problems such as deorphanization of a new therapeutic target or target identification of a drug candidate arising from phenotypic screens require large-scale predictions across the protein and molecule spaces. DTI prediction heavily relies on supervised learning algorithms that use known DTIs to learn associations between molecule and protein features, allowing for the prediction of new interactions based on learned patterns. The algorithms must be broadly applicable to enable reliable predictions, even in regions of the protein or molecule spaces where data may be scarce. In this paper, we address two key challenges to fulfill these goals: building large, high-quality training datasets and designing prediction methods that can scale, in order to be trained on such large datasets. First, we introduce LCIdb, a curated, large-sized dataset of DTIs, offering extensive coverage of both the molecule and druggable protein spaces. Notably, LCIdb contains a much higher number of molecules than publicly available benchmarks, expanding coverage of the molecule space. Second, we propose Komet (Kronecker Optimized METhod), a DTI prediction pipeline designed for scalability without compromising performance. Komet leverages a three-step framework, incorporating efficient computation choices tailored for large datasets and involving the Nyström approximation. Specifically, Komet employs a Kronecker interaction module for (molecule, protein) pairs, which efficiently captures determinants in DTIs, and whose structure allows for reduced computational complexity and quasi-Newton optimization, ensuring that the model can handle large training sets, without compromising on performance. Our method is implemented in open-source software, leveraging GPU parallel computation for efficiency. We demonstrate the interest of our pipeline on various datasets, showing that Komet displays superior scalability and prediction performance compared to state-of-the-art deep learning approaches. Additionally, we illustrate the generalization properties of Komet by showing its performance on an external dataset, and on the publicly available LH benchmark designed for scaffold hopping problems. Komet is available open source at https://komet.readthedocs.io and all datasets, including LCIdb, can be found at https://zenodo.org/records/10731712.}
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

`MIT License <LICENSE>`
