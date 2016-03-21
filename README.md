# D*SCRIPT

D\*Script is Lab41's project for exploring offline writer identification in handwritten documents. The repository includes:

* code for preparing data from several commonly available datasets
* implementations of several algorithms for feature extraction and classification
* resources for evaluating identification and retrieval perofrmance according to standard metrics
* a functioning AngularJS application for visualizing handwriting corpora and document similarity

## Where to start

*d-script/dockers* contains several Dockerized development environments for doing deep learning in Keras. If you want to take advantage of our CNN feature extractors or convolutional denoising autoencoder and do not hae your own deep learning development environment, we recommend modifying one of these to suit your development needs.

*d-script/denoiser* contains most of what you need to assemble and train our convolution denoising autoencoder, LiNet.

*d-script/evaluation* has Python code and Jupyter notebooks implementing several evaluation metrics for writer identification.

*d-script/globalclassify* contains some code for deep feature extractors in addition to nearest-neighbors classification.

*d-script/ui* has a web frontend for visualizing document classification and writer identification

## Further information

Check out [http://www.lab41.org/] for more information on the lab, and watch the Lab blog [http://www.lab41.org/gab41] for more on D\*Script and the performance of the identification algorithms we used.


