# Explain about file/folder:
* 1. dog_cat folder: dataset with 2500 cat images, 2500 dog images
* 2. extract_features.py: using ResNet50 to extract feature vectors
* 3. hdf5datasetwriter.py: export feature vectors to file .hdf5
* 4. setup.py: run extract_features.py, hdf5datasetwriter.py, tsne, plot
* 5. features.hdf5: save feature vectors
* 6. features_by_tsne.npy: save feature vectors after reducing dimensions by t-sne
* 7. visualize_feature_vectors.png: save the plot of t-sne feature vectors
