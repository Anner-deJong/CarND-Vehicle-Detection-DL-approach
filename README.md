# This repository is a revisit of:
# **CarND Project 5 ** 
# **Vehicle Detection** 

## Description

**In the [original Udacity project](https://github.com/Anner-deJong/Self-Driving-Car/tree/master/CarND-Vehicle-Detection) I used a computer vision approach to detect vehicles on a highway, with [this result](https://drive.google.com/open?id=1_czpQYQxwkScnPqkoOtQgtBYlUTEc3fT).** <br>
**The detection is okay, but not splendid. That's why right now I am revisiting this project, to see if I can improve on the prior result with a DL approach. First I planned to train my own model, but with limited computing power, I instead opted for the following:** <br>
**I found an amazing tutorial from [Ayoosh Kathuria](https://blog.paperspace.com/tag/series-yolo/) that guides you in coding up a YOLO v3 detection model in PyTorch from almost scratch, only using this official YOLO v3 [config file](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg).** <br>
**It is a pity the tutorial doesn't include the steps how to train the model, maybe I will try to build this later myself. However, simply loading in the official [weights <- whatch out, download link](https://pjreddie.com/media/files/yolov3.weights) seems to give high enough accuracy to directly apply to this project. (The weights are trained on the COCO dataset, including a car class - which is all we need).** <br>

---

## Set up and prerequisites

This repository was built with Python 3.6 and PyTorch 0.4.1 and opencv 3.4.2. It also uses matplotlib (3.0.0) for plotting the results. Other than that, it is extremely simple to get the YOLO model up and running:
Simply copy the '_cfg_weights_utils_' folder and the '_darknet.py_' file to your directory. You will have to download the [weights](https://pjreddie.com/media/files/yolov3.weights) and put them in the correct folder, et voila.

Running inference can be done through a wrapper class YOLOv3, as can be seen in the notebook. The input for `.inference()` can be a freshly read image from disk, or otherwise should be a raw numpy image, BGR, int values in [0, 255]:

    from darknet import *
    import import cv2
    
    yolo = YOLOv3()
    img     = cv2.imread('PATH_TO_IMAGE_FILE')  # or get another image input
    ann_img = yolo.inference(img)               # returning the annotated image
    
If this is too much abstraction and you would like the raw detections, please take a look at the YOLOv3 class and other functions in '_darknet.py_'.

The basic overall structure (in pseudo code) is:

    class `Darknet`
        initialization:
        `parse_cfg(config_file)`: read the config file and parse this into meaningful dictionaries containing all the information about the model architecture.
        `create_modules()`: build the architecture in PyTorch
        
    member function:
        load_weights(): load in pretrained weights
        `forward()`: implements a forward inference pass through the network and returns predictions
        
    Outside of the class:
        get_detections(pred, obj_confidence, nms_confidence, class_names): thresholding and Non Maximum Suppression of the class's predictions.
        
If you would like to go really deep, I also suggest you read the [blogs from Ayoosh Kathuria](https://blog.paperspace.com/tag/series-yolo/).

## YOLOv3 PyTorch

An immense thanks to Ayoosh Kathuria for his tutorial. There is no way I could've build all this up without his blogs. He has his own version of the model repository which can be found [here](https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch). <br>
That said, this may look like a copy, but it is not. Doing this project as a PyTorch exercise for myself, simply transcribing all the code wouldn't make sense. <br>
Therefore, my general approach was: read the descriptions, interpret, code, and only then check with his code. As a result,
the YOLO v3 part of this repository (all but the notebook) has the same structure as the Ayoosh Kathuria's, and performs the same preprocessing, inference and post-processing on an input image. <br>
However, many functions are implemented with either different syntax, taking/returning differently stored data, or simply following a different line of reasoning. Some functions might be better (I actually made a few small pull requests), some might be worse. I did not implement timing in order to compare.

### Inference, no training

Currently the yolo model code allows for inference on a single image at a time. This is all we needed for the video, but it would be nice in the future to have batch capabilities (partly already implemented) or training capabilities.

### other room for improvements

Some possible improvements include (non-exhaustive):

* Non Maximum Suppression function can be optimized (IoU can be done in parallel with torch matrix operations for example. Perhaps this [link](https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/) has some more useful improvements.
* The `YOLOv3` class is very high level and as such very limited in use case. Opening it up a little bit could provide a wider use case of the current repository.
* The tutorial includes a steps for a command line argument parser, which is not too hard to implement once desired
* The tutorial also prescribes to clip bounding boxes to the border of the image. I am not sure however if I agree with that and left the bounding boxes 'unbounded'
* Text annotation sizes are currently hardcoded, they might not work for different kinds of image input dimensions, and hence they should be dynamic
* All the hyperparameters of running inference (image resolution, threshold etc.) are currently also abstracted away a bit too much perhaps

## Video result

[The newly annotated video can be found here](https://drive.google.com/file/d/1qYRIZ3PHJpytPzutnPq0xabH6oRUtDmJ/view?usp=sharing)

[(again, the old video)](https://drive.google.com/file/d/1_czpQYQxwkScnPqkoOtQgtBYlUTEc3fT/view)

## Discussion

Checking the difference between the traditional CV approach video, and this DL video, it is once again stunning how DL trumps the results. Especially with environment conditions where the CV approach has some trouble (different lighting, shadows, different color tarmac) or with some overlap, the DL approach seems to have no problem at all.

The only problem DL seems to have, especially at the end of the video, is a double detection of a car as both a car _and_ a truck. 












