wget http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
wget http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
wget https://github.com/BVLC/caffe/raw/master/python/caffe/imagenet/ilsvrc_2012_mean.npy
wget https://dl.dropboxusercontent.com/u/38822310/gender_net.caffemodel
wget https://dl.dropboxusercontent.com/u/38822310/age_net.caffemodel
wget http://places.csail.mit.edu/model/googlenet_places205.tar.gz
tar xzvf googlenet_places205.tar.gz
mv googlenet_places205/googlelet_places205_train_iter_2400000.caffemodel .
mv googlenet_places205/categoryIndex_places205.csv .
