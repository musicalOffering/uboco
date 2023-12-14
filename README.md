"UBoCo : Unsupervised Boundary Contrastive Learning
for Generic Event Boundary Detection"
=============
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Kang_UBoCo_Unsupervised_Boundary_Contrastive_Learning_for_Generic_Event_Boundary_Detection_CVPR_2022_paper.pdf)


Training and Testing the model
=============
1. Prepare data in the dataset folder. Here is the annotation files:   
[k400_mr345_train_min_change_duration0.3.pkl](https://drive.google.com/file/d/1qSSgFFPHOr6Cgzkm0Ytfc6IC49AWE6az/view?usp=drive_link),   [k400_mr345_val_min_change_duration0.3.pkl](https://drive.google.com/file/d/1dM9Bq6o4H0gkHy-dtkKE1go4eVlyh_YC/view?usp=drive_link)   
So, the file tree should be like this:   
    ```
    data
    |--resnet50_padded
    |  |--k400_mr345_train_min_change_duration0.3.pkl
    |  |--k400_mr345_val_min_change_duration0.3.pkl
    |  |--train
    |  |--|--_4insWyfuuw
    |  |--|--_4oxxQeQ2aM
    |  |--|--...
    |  |--val
    |  |--|--01BFInmg3Zs
    |  |--|--77aDh42ddw8
    |  |--|--...
    |  |--train_of_train
    |  |--|--...
    |  |--...
    ```    
    Due to the large size of the features, approximately 50GB, we are unable to offer a direct download link. For details on the extraction configuration, please refer to our paper.
    Please ensure that all videos are uniformly padded to achieve the same length. This means that each video should have a consistent tensor shape, specifically [length, feature_dim] = [40, 2048].
    
    The `DATASET_MODE` setting in config.py defines how the dataset is divided. Initially, you should divide the entire dataset into training and validation (val) subsets, in accordance with the official annotations. If you set `DATASET_MODE == "tv"`, you are working with this split strategy. Optionally, you can further split the training subset into two parts: 'train_of_train' and 'train_of_val'. If you set `DATASET_MODE == "tvt"`, the 'train_of_val' serves as the validation subset, and the original validation (val) segment is used solely for testing purposes.

2. Copy some files to `resnet_visualizing`, or `visualizing` folder, and update the `VAL_VIDEOS` in `config.py`. This is for the visualization during training procedure.

3. Just run `main.py`. It contains both training and testing codes.

