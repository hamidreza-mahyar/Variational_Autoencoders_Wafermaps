# Unsupervised Wafermap Patterns Clustering via Variational Autoencoders

This is the code produced as part of the paper _Unsupervised Wafermap Patterns Clustering via Variational Autoencoders_ 

In this paper, we presented a machine learning approach for unsupervised clustering of spatial patterns in wafermap measurement data. Measured test values are first pre-processed using some computer vision techniques, followed by a feature extraction based on variational autoencoders to decompose high-dimensional wafermaps into a low-dimensional latent representation. Final step is to detect the structure of this latent space and assign individual wafers into clusters.

> "Unsupervised Wafermap Patterns Clustering via Variational Autoencoders",
> Peter Tulala*, Hamidreza Mahyar*, Elahe Ghalebi, Radu Grosu. in Proc The International Joint Conference on Neural Networks (IJCNN), Rio de Janeiro, Brazil, July 2018, pp. 1-8. (*Authors contributed equally)(Oral presentation) [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/8489422)

## Packages needed

 - `jupyter`
 - `scipy`
 - `scikit-image`
 - `keras`
 - `tensorflow`
 - `pandas`
 - `tqdm`
 - `seaborn`
 - `np_utils`
 - `sklearn`
 - `gmm-mml`

## Experiment execution

Running experiments:

### `Wafermaps_PreProcessing.ipynb`

This notebook in `notebooks` folder is used for pre-proccesing of semiconductor wafermaps dataset. Data pre-processing is a crucial step addressing several data quality issues before applying the machine learning algorithm. The main goal of this step is to ensure that individual wafer measurements are comparable.  The result of data pre-processing step is a cleansed dataset that can be used for further feature extraction and clustering tasks.

### `Wafermaps_Variational_AutoEncoders.ipynb`

This notebook in `notebooks` folder is used for extracting meaningful information from the cleansed wafermap dataset,  based on auto-encoding variational Bayes. A variational autoencoder is trained on cleansed wafermaps to extract latent features. Then, we apply k-means clustering on latent features to group wafers based on a distant measure.

## Docker

Please check `run_jupyter.sh` file.
