# ComputerVisionSummarization
This repository focus on the summary of computer vision

# Video collection
## Link
https://pan.baidu.com/s/1eSIVG90

## video rank
- holoportation_ virtual 3D teleportation in real-time (Microsoft Research).mp4
- Realtime Multi-Person 2D Human Pose Estimation using Part Affinity Fields, CVPR 2017 Oral
- Full-Resolution Residual Networks (FRRNs) for Semantic Image Segmentation in Street Scenes
- YOLO v2
- DeepGlint CVPR2016

# Detection

# Sementation
## mask rcnn
The mask rcnn is proposed by [KaiMing](https://arxiv.org/abs/1703.06870), and implied in github [repostory](https://github.com/matterport/Mask_RCNN.git)

- mask rcnn extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box  recognition. 

- output:
    - a class label
    - a bounding-box offset
    - object mask

- It can run at 5 fps and training on COCO takes one to two days on a single 8-GPU machine. 

- It has another application: human pose estimation, instance segementation, bounding-box object detection, and person keypoint detection, camera calibration. 
    - By viewing each keypoint as a one-hot binary mask, it can estimate human pose.

- It belongs to the instance segmentation field.

I have the mask rcnn in bus scene. 

![](./images/mask-rcnn-result.png)
 
It performs well. This is all the [result](https://pan.baidu.com/s/1nvefTPZ)

# Lip language
It's an amazing thing that training lip language recogition

## Learning Lip Sync from Audio
- given the audio of President Barack Obama, we synthesize a high quality video of him speaking with accurate lip sync. see [video](https://youtu.be/9Yq67CjDqvw)

# Some Interesting
## Aging photo prediction
- takes a single photograph of a child as input and automatically produces a series of age-progressed outputs between 1 and 80 years of age, accounting for pose, expression, and illumination. see [video](https://youtu.be/QuKluy7NAvE?list=PLDeWtkr3kw-2_888O0qgVNC-uoEyyaWmm)

## 3D panorama
- [A 3D panorama](https://youtu.be/1oWBsR8zTP0)

## [PanoCatcher](https://youtu.be/DCcjgZmDwJ0)

# Tracking

# Action Recognition

# 3D Detection

# 3D Sementation

# 3D Action Recognition

# 3D Point Cloud Representation


# Summary
- Mask RCNN is amazing, but it's not fast enough for real time detection.
- There are lots of computer vision tasks need to be done, and only few tasks are finished. Obejct recognition is the simplest task, which is extremly handled and the rate of recognition is more than that of human beings. But, the majority tasks are still need to be done, such as: action recogition, action predict, 3D object recognition, 3D object representation, 3D action recognition, represention of speak, smell, feel and vision. Machine vision is the kernel task for robot intelligence. So don't worry about nothing to do in this field.

# Reference
http://www.themtank.org/a-year-in-computer-vision
