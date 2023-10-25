The code in this repository implements the "BiG-FSLF: A Cross Heterogeneous Domain Few-Shot Learning Framework Based on Bidirectional Generation for Hyperspectral Image Change Detection".

We run this code with python 3.8, torch==1.11.0, and CUDA==11.3.

Please kindly cite our paper if this code is useful and helpful for your research.

X. Wang, S. Li, X. Zhao, K. Zhao, "BiG-FSLF: A Cross Heterogeneous Domain Few-Shot Learning Framework Based on Bidirectional Generation for Hyperspectral Image Change Detection," in IEEE Transactions on Geoscience and Remote Sensing, 2023, doi:10.1109/TGRS.2023.3292249.

If you encounter any bugs while using this code, please do not hesitate to contact us.

First, you need to run the script MSI157_matpickle.py to generate preprocessed source domain data.
Furthermore, if you want to use your own dataset, please note that the label information of changed class and unchanged class in the ground-truth data of the source domain and target domain should be consistent.

You can also download the VHRI and HSI datasets used in our experiment from the link: 
https://pan.baidu.com/s/1k5C35y_1d0Pf3fDg17fOFA
The corresponding password is：tzud
