# Generalizing Monocular 3D Human Pose Estimation in the Wild

This repository is the implementation of the work presented in:  

>**Luyang Wang, Yan Chen**, Zhenhua Guo, Keyuan Qian, Mude Lin, Hongsheng Li, **[Jimmy S. Ren](http://www.jimmyren.com/)**,
**Generalizing Monocular 3D Human Pose Estimation in the Wild.([International Conf. on Computer Vision - Workshop on Geometry Meets Deep Learning 2019](https://arxiv.org/pdf/1904.05512.pdf))
[Watch Our Video on YouTube.](https://youtu.be/bGpP_S5fVWc)**

<p align="center">
  <img src="https://github.com/llcshappy/Give-3D-Label-in-the-Wild/blob/master/demo/3DGen.png" width="800" title="img1">
</p>

### Dependencies
>**[Tensorflow](https://www.tensorflow.org) >= 1.4.1<br>
[Pytorch](https://pytorch.org/) >= 0.3.1<br>
Numpy = 1.14.3<br>
CV2 = 3.4.1<br>**

### Dataset
You can download our processed datasets in the list. We train the **3D Label Generator** with ***Human3.6M dataset and Unity dataset***. In addition, We train the **Baseline Network** with ***MPII/LSP/AIChallenger/Human3.6M datasets***. Note that we provided the ***MPII/LSP/AIChallenger/Human3.6M datasets with high-quality 3D labels***, available through **Google Drive**. 

> - [x] **[MPII](https://drive.google.com/open?id=1joCbigfr9QbpTuvP6cklZOpW2lUR0Qsf)**
> - [x] **[LSP](https://drive.google.com/open?id=1joCbigfr9QbpTuvP6cklZOpW2lUR0Qsf)**
> - [x] **[AIChallenger](https://drive.google.com/drive/folders/1sXa6zI-2Hv8337Ur0Njfqwgzo8gjL0K4?usp=sharing)**
> - [x] **[Human3.6M](https://drive.google.com/open?id=1NEevs6Z_HHnuQ3ikSPMd_kj-2VTm3DH5)**
> - [x] **[Unity](https://drive.google.com/open?id=1wp2Ba9X4jfWXugdaZE3AU_i2WLKy8bjv)**
> - [x] **[LeftRight2D](https://drive.google.com/open?id=1uD17eDuNWQNwfMUd9CPfwsDwaqrFKJcJ)**
#### Guidelines
Download the datasets. All the compressed files suffixes are ***tar.gz***.

    tar -zxvf xxx.tar.gz
    
See more details [here](https://github.com/llcshappy/Monocular-3D-Human-Pose/blob/master/Dataset/readme.md).
### Pre-trained Model
We also provide a model pre-trained on 3D Label Generator and Baseline Network, available through [Baidu Cloud](https://pan.baidu.com/download).
- [x] **[3D Label Generator](https://drive.google.com/file/d/11SEVXOqvQuqAoUuu6CR6Qb-M77mtZdBw/view?usp=sharing)**


## Installation
Clone this repository and download our processed datasets.  

    git clone https://github.com/llcshappy/Monocular-3D-Human-Pose.git
    
## Useage

### 3D Label Generator
The code of 3D Label Generator was tested with Anaconda Python3.6 and Tensorflow. After install Anaconda and Tensorflow:
#### Step 1. Open the 3DLabelGen folder:
    
    cd 3DLabelGen/
    
#### Step2. Training Stereoscopic View Synthesis Subnetwork 
>You need to generate the right-view 2D pose.
    
    python2 gen_right.py
    
>Train the subnetwork

    ./left2right.sh
    
#### Step3. Training 3D Pose Reconstruction Subnetwork

>Train the subnetwork
    
    ./3DPose.sh

#### Step4. Geometric Search Scheme 
>See more details of the geometric search scheme in our paper. Please input the action in script 'search_h36m.py'
    
    # Input the action here
    action = 'WalkTogether'
    
>Then run this script.
    
    python2 search_h36m.py
    

#### Quick Demo
You can run the following code to see the quick demo of the **3D Label Generator**.

    ./demo.sh
    

#### Quick Demo
You can run the following code to see the quick demo of **our trained Baseline Network**.

    ./demo.sh
    


## Visualization 

<p align="center">
  <img src="https://github.com/llcshappy/Give-3D-Label-in-the-Wild/blob/master/demo/1480.jpg" width="200" title="img1">
  <img src="https://github.com/llcshappy/Give-3D-Label-in-the-Wild/blob/master/demo/165.jpg" width="200" title="img2">
  <img src="https://github.com/llcshappy/Give-3D-Label-in-the-Wild/blob/master/demo/1659.jpg" width="200" title="img3">
  <img src="https://github.com/llcshappy/Give-3D-Label-in-the-Wild/blob/master/demo/1709.jpg" width="200" title="img4">
  <img src="https://github.com/llcshappy/Give-3D-Label-in-the-Wild/blob/master/demo/1843.jpg" width="200" title="img1">
  <img src="https://github.com/llcshappy/Give-3D-Label-in-the-Wild/blob/master/demo/1988.jpg" width="200" title="img2">
  <img src="https://github.com/llcshappy/Give-3D-Label-in-the-Wild/blob/master/demo/831.jpg" width="200" title="img3">
  <img src="https://github.com/llcshappy/Give-3D-Label-in-the-Wild/blob/master/demo/86.jpg" width="200" title="img4">
  <img src="https://github.com/llcshappy/Give-3D-Label-in-the-Wild/blob/master/demo/1287.jpg" width="200" title="img1">
  <img src="https://github.com/llcshappy/Give-3D-Label-in-the-Wild/blob/master/demo/1676.jpg" width="200" title="img2">
  <img src="https://github.com/llcshappy/Give-3D-Label-in-the-Wild/blob/master/demo/1971.jpg" width="200" title="img3">
  <img src="https://github.com/llcshappy/Give-3D-Label-in-the-Wild/blob/master/demo/1998.jpg" width="200" title="img4">
</p>

## Citation

@article{wang2019generalizing,<br>
  title={Generalizing Monocular 3D Human Pose Estimation in the Wild},<br>
  author={Wang, Luyang and Chen, Yan and Guo, Zhenhua and Qian, Keyuan and Lin, Mude and Li, Hongsheng and Ren, Jimmy S},<br>
  journal={arXiv preprint arXiv:1904.05512},<br>
  year={2019}<br>
}<br>


