wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
tar xzvf 101_ObjectCategories.tar.gz
python make_train_data.py 101_ObjectCategories
mv images _images
cd _images
python ../resize.py *
mv resized ../images
cd ..
python compute_mean.py train.txt
