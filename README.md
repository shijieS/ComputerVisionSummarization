<!-- TOC -->

- [1. CVPR Paper](#1-cvpr-paper)
    - [1.1. All CVPR2018 Paper](#11-all-cvpr2018-paper)
        - [1.1.1. Tracking](#111-tracking)
- [2. Video collection](#2-video-collection)
    - [2.1. Link](#21-link)
    - [2.2. video rank](#22-video-rank)
- [3. Detection](#3-detection)
- [4. Sementation](#4-sementation)
    - [4.1. mask rcnn](#41-mask-rcnn)
- [5. Lip language](#5-lip-language)
    - [5.1. Learning Lip Sync from Audio](#51-learning-lip-sync-from-audio)
- [6. Some Interesting](#6-some-interesting)
    - [6.1. Aging photo prediction](#61-aging-photo-prediction)
    - [6.2. D panorama](#62-d-panorama)
    - [6.3. PanoCatcher](#63-panocatcher)
- [7. Tracking](#7-tracking)
    - [7.1. MOT](#71-mot)
    - [7.2. Correlation Filter](#72-correlation-filter)
    - [7.3. End-to-end representation learning for correlation filter based tracking](#73-end-to-end-representation-learning-for-correlation-filter-based-tracking)
    - [7.4. Attentional Correlation Filter Network for Adaptive Visual Tracking](#74-attentional-correlation-filter-network-for-adaptive-visual-tracking)
    - [7.5. Context-Aware Correlation Filter Tracking](#75-context-aware-correlation-filter-tracking)
- [8. Action Recognition](#8-action-recognition)
- [9. Reconstruction](#9-reconstruction)
- [10. Detection](#10-detection)
- [11. Sementation](#11-sementation)
- [12. Action Recognition](#12-action-recognition)
- [13. Point Cloud Representation](#13-point-cloud-representation)
- [14. Summary](#14-summary)
- [15. Reference](#15-reference)

<!-- /TOC -->


# 1. CVPR Paper
All the paper is available at [official website](http://cvpr2018.thecvf.com/program/main_conference).

The offline list of paper is available at [this](./papers/cvpr2018-paper-list.csv)

## 1.1. All CVPR2018 Paper
### 1.1.1. Tracking

| Paper ID | Type      | Title                                                                                                                                          |
| -------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| 122      | Poster    | [Detect-and-Track: Efficient Pose Estimation in Videos](./papers/1712.09184.pdf)                                                               |
| 255      | Poster    | Multi-Cue Correlation Filters for Robust Visual Tracking                                                                                       |
| 281      | Spotlight | Tracking Multiple Objects Outside the Line of Sight using Speckle Imaging                                                                      |
| 281      | Poster    | Tracking Multiple Objects Outside the Line of Sight using Speckle Imaging                                                                      |
| 369      | Oral      | [Total Capture: A 3D Deformation Model for Tracking Faces, Hands, and Bodies](./papers/1801.01615.pdf)                                         |
| 369      | Poster    | Total Capture: A 3D Deformation Model for Tracking Faces, Hands, and Bodies                                                                    |
| 423      | Spotlight | Fast and Accurate Online Video Object Segmentation via Tracking Parts                                                                          |
| 423      | Poster    | Fast and Accurate Online Video Object Segmentation via Tracking Parts                                                                          |
| 678      | Poster    | [Learning Attentions: Residual Attentional Siamese Network for High Performance Online Visual Tracking](./papers/CVPR18RASTrackCameraV3.3.pdf) |
| 736      | Spotlight | [GANerated Hands for Real-Time 3D Hand Tracking from Monocular RGB](./papers/1712.01057.pdf)                                                   |
| 736      | Poster    | GANerated Hands for Real-Time 3D Hand Tracking from Monocular RGB                                                                              |
| 890      | Poster    | [CarFusion: Combining Point Tracking and Part Detection for Dynamic 3D Reconstruction of Vehicles](./papers/CarFusion.pdf)                     |
| 892      | Poster    | [Context-aware Deep Feature Compression for High-speed Visual Tracking](./papers/1803.10537.pdf)                                               |
| 1022     | Poster    | A Benchmark for Articulated Human Pose Estimation and Tracking                                                                                 |
| 1194     | Poster    | [Hyperparameter Optimization for Tracking with Continuous Deep Q-Learning](./papers/Q-learning.pdf)                                            |
| 1264     | Poster    | [End-to-end Flow Correlation Tracking with Spatial-temporal Attention](./papers/1711.01124.pdf)                                                |
| 1280     | Spotlight | [VITAL: VIsual Tracking via Adversarial Learning](./papers/1804.04273.pdf)                                                                     |
| 1280     | Poster    | VITAL: VIsual Tracking via Adversarial Learning                                                                                                |
| 1304     | Poster    | SINT++: Robust Visual Tracking via Adversarial Hard Positive Generation                                                                        |
| 1353     | Poster    | [Learning Spatial-Temporal Regularized Correlation Filters for Visual Tracking](./papers/1803.08679.pdf)                                       |
| 1439     | Poster    | [Efficient Diverse Ensemble for Discriminative Co-Tracking](./papers/1711.06564.pdf)                                                           |
| 1494     | Poster    | [Correlation Tracking via Joint Discrimination and Reliability Learning](./papers/cvpr2018_correlation_tracking.pdf)                           |
| 1676     | Spotlight | [Learning Spatial-Aware Regressions for Visual Tracking](./papers/1706.07457.pdf)                                                              |
| 1676     | Poster    | Learning Spatial-Aware Regressions for Visual Tracking                                                                                         |
| 1679     | Poster    | Fusing Crowd Density Maps and Visual Object Trackers for People Tracking in Crowd Scenes                                                       |
| 1949     | Poster    | Rolling Shutter and Radial Distortion are Features for High Frame Rate Multi-camera Tracking                                                   |
| 2129     | Poster    | High-speed Tracking with Multi-kernel Correlation Filters                                                                                      |
| 2628     | Poster    | [A Causal And-Or Graph Model for Visibility Fluent Reasoning in Tracking Interacting Objects](./papers/1709.05437.pdf)                         |
| 2951     | Spotlight | High Performance Visual Tracking with Siamese Region Proposal Network                                                                          |
| 2951     | Poster    | High Performance Visual Tracking with Siamese Region Proposal Network                                                                          |
| 3013     | Oral      | Fast and Furious: Real Time End-to-End 3D Detection, Tracking and Motion Forecasting with a Single Convolutional Net                           |
| 3013     | Poster    | Fast and Furious: Real Time End-to-End 3D Detection, Tracking and Motion Forecasting with a Single Convolutional Net                           |
| 3292     | Spotlight | MX-LSTM: mixing tracklets and vislets to jointly forecast trajectories and head poses                                                          |
| 3292     | Poster    | MX-LSTM: mixing tracklets and vislets to jointly forecast trajectories and head poses                                                          |
| 3502     | Poster    | A Prior-Less Method for Multi-Face Tracking in Unconstrained Videos                                                                            |
| 3583     | Poster    | [Towards dense object tracking in a 2D honeybee hive](./papers/1712.08324.pdf)                                                                 |
| 3817     | Spotlight | [Good Appearance Features for Multi-Target Multi-Camera Tracking](./papers/1709.07065.pdf)                                                     |
| 3817     | Poster    | Good Appearance Features for Multi-Target Multi-Camera Tracking                                                                                |
| 3980     | Poster    | [A Twofold Siamese Network for Real-Time Object Tracking](./papers/1802.08817.pdf)                                                             |


# 2. Video collection
## 2.1. Link
https://pan.baidu.com/s/1eSIVG90

## 2.2. video rank
- holoportation_ virtual 3D teleportation in real-time (Microsoft Research).mp4
- Realtime Multi-Person 2D Human Pose Estimation using Part Affinity Fields, CVPR 2017 Oral
- Full-Resolution Residual Networks (FRRNs) for Semantic Image Segmentation in Street Scenes
- YOLO v2
- DeepGlint CVPR2016

# 3. Detection

# 4. Sementation
## 4.1. mask rcnn
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

# 5. Lip language
It's an amazing thing that training lip language recogition

## 5.1. Learning Lip Sync from Audio
- given the audio of President Barack Obama, we synthesize a high quality video of him speaking with accurate lip sync. see [video](https://youtu.be/9Yq67CjDqvw)

# 6. Some Interesting
## 6.1. Aging photo prediction
- takes a single photograph of a child as input and automatically produces a series of age-progressed outputs between 1 and 80 years of age, accounting for pose, expression, and illumination. see [video](https://youtu.be/QuKluy7NAvE?list=PLDeWtkr3kw-2_888O0qgVNC-uoEyyaWmm)

## 6.2. D panorama
- [A 3D panorama](https://youtu.be/1oWBsR8zTP0)

## 6.3. [PanoCatcher](https://youtu.be/DCcjgZmDwJ0)

# 7. Tracking

## 7.1. MOT

| method name | title                                                                                                                                                | paper   | author                                                             | rate        |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ------------------------------------------------------------------ | ----------- |
| CDA_DDALv2  | [Confidence-Based Data Association and Discriminative Deep Appearance Learning for Robust Online Multi-Object Tracking](./papers/07893777.pdf)       | TPAMI   | MLA Bae, Seung-Hwan, and Kuk-Jin Yoon.                             | reading now |
| FWT         | [Fusion of Head and Full-Body Detectors for Multi-Object Tracking](./papers/1705.08314.pdf)                                                          | CVPR18  | Roberto Henschel, Laura Leal-Taixe, Daniel Cremers, Bodo Rosenhahn | reading now |
| LMP         | [Multiple people tracking by lifted multicut and person re-identification](./papers/Multiple_People_Tracking.pdf)                                    | CVPR17  | Tang, Siyu, et al.                                                 | reading now |
| NLLMPa      | [Joint graph decomposition & node labeling: Problem, algorithms, applications.](./papers/1611.04399.pdf)                                             | CVPR17  | Levinkov, Evgeny, et al.                                           | reading now |
| QuadMOT16   | [ Multi-Object Tracking with Quadruplet Convolutional Neural Networks](./papers/08099886.pdf)                                                        | CVPR17  | Son, Jeany, et al.                                                 | reading now |
| EDMT        | [Enhancing Detection Model for Multiple Hypothesis Tracking](./papers/Chen_Enhancing_Detection_Model)                                                | CVPR17w | Chen, Jiahui, et al.                                               | reading now |
| AMIR        | [Tracking the untrackable: Learning to track multiple cues with long-term dependencies](./papers/1701.01909.pdf)                                     | ICCV17  | Sadeghian, Amir, Alexandre Alahi, and Silvio Savarese.             | reading now |
| STAM16      | [Online Multi-Object Tracking Using CNN-based Single Object Tracker with Spatial-Temporal Attention Mechanism.](./papers/1708.02843.pdf)             | ICCV17  | Chu, Qi, et al.                                                    | reading now |
| LINF1       | [Improving Multi-Frame Data Association with Sparse Representations for Robust Near-Online Multi-Object Tracking](./papers/978-3-319-46484-8_47.pdf) | ECCV16  | L. Fagot-Bouquet, R. Audigier, Y. Dhome, F. Lerasle                | reading now |
| EAMTT       | [Multi-target tracking with strong and weak detections](./papers/eamtt.pdf)                                                                          | ECCV16w | R. Sanchez-Matilla, F. Poiesi, A. Cavallaro                        | reading now |
| LTTSC-CRF   | [Long-Term Time-Sensitive Costs for CRF-Based Tracking by Detection](./papers/lttsc-crf.pdf)                                                         | ECCV16w | Le, Nam, Alexander Heili, and Jean-Marc Odobez.                    | reading now |

## 7.2. Correlation Filter
## 7.3. End-to-end representation learning for correlation filter based tracking
> It is a tracking method based on deep learning. This author designed a network consisting of correlation filter layer, who solved the backpropagation program 
- I have tried this method. But it doesn't work well and have some test failure cases, as following

![](./images/CFNetFailCase.jpg)  
- abstract
    We present a framework that allows the explicit incorporation of global context within CF trackers. We reformulate the original optimization problem and provide a closed form solution for single and multidimensional features in the primal and dual domain.
- *video*, [paper](https://arxiv.org/abs/1704.06036v1), [matlab code](https://github.com/bertinetto/cfnet.git), [python code](https://github.com/torrvision/siamfc-tf)
- advantage:
    - It's an end-to-end tracking method, which can be trained directly.
    - It can run in real-time.
- disadvantage:
    - It's will drift with the object occlusion
    - It's will scale wrongly with the object enlarge or being small.
    
- My opion:
    - Tracking should be combined both the object feature itself and the context feature.
 
## 7.4. Attentional Correlation Filter Network for Adaptive Visual Tracking
- paper, [video](https://youtu.be/WCcaxLiDuyI), [code](https://github.com/jongwon20000/ACFN.git)
- advantage:
- disadvantage:
    - slow. Cannot run in real-time.
 
## 7.5. Context-Aware Correlation Filter Tracking
- Bas

# 8. Action Recognition
- [video](https://youtu.be/pW6nZXeWlGM), [paper](https://www.youtube.com/redirect?q=https%3A%2F%2Farxiv.org%2Fabs%2F1611.08050&event=video_description&v=pW6nZXeWlGM&redir_token=Es81J58cRaIs9GIo7M9nO-sakuB8MTUxNDAxNDIwMUAxNTEzOTI3ODAx)

# 9. Reconstruction
- [video](https://youtu.be/z_NJxbkQnBU)
- [paper](https://arxiv.org/pdf/1707.06375.pdf), [video](https://youtu.be/uf4-l6h7iGM), 
- [video](https://youtu.be/2CvFHy5jk1c), [paper](http://www.frc.ri.cmu.edu/~syang/Publications/icra_2016.pdf), [code](http://www.frc.ri.cmu.edu/~syang/corridor_pop_up.html)
# 10. Detection

# 11. Sementation

# 12. Action Recognition

# 13. Point Cloud Representation


# 14. Summary
- Mask RCNN is amazing, but it's not fast enough for real time detection.
- There are lots of computer vision tasks need to be done, and only few tasks are finished. Obejct recognition is the simplest task, which is extremly handled and the rate of recognition is more than that of human beings. But, the majority tasks are still need to be done, such as: action recogition, action predict, 3D object recognition, 3D object representation, 3D action recognition, represention of speak, smell, feel and vision. Machine vision is the kernel task for robot intelligence. So don't worry about nothing to do in this field.

# 15. Reference
http://www.themtank.org/a-year-in-computer-vision
