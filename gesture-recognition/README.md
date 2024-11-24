# Gesture Recognition

>[!note]
>if the container is crashing due to a QT error, ensure X11 fowarding is enabled on the host by running:
>
>```bash
>xhost +local:
>```

## Model

![model](model.png)

## Usage

### gestures

![gestures](gestures.jpg)

## directory structure

### app.py

This is a sample program for inference.

In addition, learning data (key points) for hand sign recognition,

You can also collect training data (index finger coordinate history) for finger gesture recognition.

### keypoint_classification.ipynb

This is a model training script for hand sign recognition.

### model/keypoint_classifier

This directory stores files related to hand sign recognition.

The following files are stored.

* Training data(keypoint.csv)
* Trained model(keypoint_classifier.tflite)
* Label data(keypoint_classifier_label.csv)
* Inference module(keypoint_classifier.py)

### utils/cvfpscalc.py

This is a module for FPS measurement.

## Training

Hand sign recognition and finger gesture recognition can add and change training data and retrain the model.

### Hand sign recognition training

#### 1.Learning data collection

Start the program with the `MODE` environment variable set to `KEYPOINT_TRAINING`.

![](https://user-images.githubusercontent.com/37477845/102235423-aa6cb680-3f35-11eb-8ebd-5d823e211447.jpg)

If you press "0" to "9", the key points will be added to "model/keypoint_classifier/keypoint.csv" as shown below.

1st column: Pressed number (used as class ID), 2nd and subsequent columns: Key point coordinates.

![csv table](https://user-images.githubusercontent.com/37477845/102345725-28d26280-3fe1-11eb-9eeb-8c938e3f625b.png)

The key point coordinates are the ones that have undergone the following preprocessing up to ④.

![keypoints](./handgestureLandMarks.png)

In the initial state, three types of learning data are included: open hand (class ID: 0), close hand (class ID: 1), and pointing (class ID: 2).

If necessary, add 3 or later, or delete the existing data of csv to prepare the training data.

<img src="https://user-images.githubusercontent.com/37477845/102348846-d0519400-3fe5-11eb-8789-2e7daec65751.jpg" width="25%">
<img src="https://user-images.githubusercontent.com/37477845/102348855-d2b3ee00-3fe5-11eb-9c6d-b8924092a6d8.jpg" width="25%">
<img src="https://user-images.githubusercontent.com/37477845/102348861-d3e51b00-3fe5-11eb-8b07-adc08a48a760.jpg" width="25%">

#### 2.Model training

Open "[keypoint_classification.ipynb](keypoint_classification.ipynb)" in Jupyter Notebook and execute from top to bottom.

To change the number of training data classes, change the value of "NUM_CLASSES = 3"

and modify the label of "model/keypoint_classifier/keypoint_classifier_label.csv" as appropriate.

#### X.Model structure

The image of the model prepared in "[keypoint_classification.ipynb](keypoint_classification.ipynb)" is as follows.
<img src="https://user-images.githubusercontent.com/37477845/102246723-69c76a00-3f42-11eb-8a4b-7c6b032b7e71.png" width="50%"><br><br>

# License

This component was build off of the work of Kazuhito Takahashi and Nikita Kiselov. The original work can be found [here](https://github.com/kinivi/hand-gesture-recognition-mediapipe) and is licensed under the [Apache v2 license](LICENSE).
