## This repository is a revisit of:
# **CarND Project 5: Vehicle Detection** 

---

## Description

**In the [original Udacity project](https://github.com/Anner-deJong/Self-Driving-Car/tree/master/CarND-Vehicle-Detection) I used a computer vision approach to detect vehicles on a highway, with [this result](https://drive.google.com/open?id=1_czpQYQxwkScnPqkoOtQgtBYlUTEc3fT).** <br>
The detection is okay, but not splendid. That's why right now I am revisiting this project, to see if I can improve on the prior result with a DL approach. <br> I First planned to train my own model, but with limited computing power, I instead opted for the following: <br>
<br>
**I found an amazing tutorial from [Ayoosh Kathuria](https://blog.paperspace.com/tag/series-yolo/) that guides you in coding up a YOLO v3 detection model in PyTorch from almost scratch, only using the official YOLO v3 config file and weights. <br>
I build the YOLO v3 model code as a separate, stand-alone [repository](https://github.com/Anner-deJong/YOLOv3-PyTorch). You will have to clone/download it and replace the empty _Yolov3_ folder in this repository. Furtyhermore, follow instructions in the Yolov3 repository about how to get the weights.** <br>
It is a pity the tutorial doesn't include the steps how to train the model, maybe I will try to build this later myself. However, the orignal weights are trained on the COCO dataset, which includes a car class - all we need for the current purpose). <br>

---

## Set up and prerequisites

This repository and the required Yolov3 repository, were built with Python 3.6, PyTorch 0.4.1 and opencv 3.4.2. This project also uses IPython and matplotlib (3.0.0) for plotting the results. Other than that, it is extremely simple to get the YOLO model up and running.
Simply clone/download/copy the [Yolov3 repository](https://github.com/Anner-deJong/YOLOv3-PyTorch) in place of the Yolov3 folder from this repository (_instead_, **not** _inside_).

The Yolov3 repo contains a very abstract class _YOLOv3( )_ that makes running inference very easy (yet so abstract that it cannot do anything else). Please see the notebook. It simply takes in an image and returns an image copy with annotated detections:

    from Yolov3 import *
    
    #instantiate a yolo model class
    yolo = YOLOv3()
    
    # run inference on an image
    img     = cv2.imread('test_video_37.jpg')
    ann_img = yolo.inference(img)
    
The input for `.inference()` can be a freshly read image from disk, or otherwise should be a raw numpy image, BGR, int values in [0, 255]. Please check the Yolov3 repository for more details or if you want less abstraction (raw detection for example).


## Video result

The notebook includes a function that automatically runs through each frame of project's video, annotates it, and saves all the resulting annotated back to disk in form of a video.

[The newly annotated video can be found here](https://drive.google.com/file/d/1qYRIZ3PHJpytPzutnPq0xabH6oRUtDmJ/view?usp=sharing)

[(again, the old video)](https://drive.google.com/file/d/1_czpQYQxwkScnPqkoOtQgtBYlUTEc3fT/view)

## Discussion

Checking the difference between the traditional CV approach video, and this DL video, it is once again stunning how DL trumps the non-DL results. Especially with environment conditions where the CV approach has some trouble (different lighting, shadows, different color tarmac) or with some overlap, the DL approach seems to have no problem at all.

The only problem DL seems to have, especially at the end of the video, is a double detection of a car as both a car _and_ a truck. 












