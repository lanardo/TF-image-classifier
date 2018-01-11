### Image classification with TF

# Dependencies
1. Anaconda 3.5 (python 3.5)

2. Tensorflow
    ``` pip install tensorflow ```

3. OpenCV 3.2
    ``` pip install opencv-contrib-python ```

4. numpy
    ``` pip install numpy ```

# Running the script
1. train with classified images
    1.1. paste the classified image dataset on the directory `./data/train/`
    1.2. run the functions of `train.py`
        ` train_data() `
        ` train() `
    the trained model would be saved on the directory `./model/`
2. classify the new images with using trained modeling
    2.1. paste the new images on the directory `./data/test/`
    2.2 run the function of `classifier.py`
        `inference("./data/test", "./model")`
        here: first arg is the directory path for the testing images
              second is the directory for trained model
