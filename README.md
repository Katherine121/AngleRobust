# sequence
# Background
In this project, we propose a novel angle robustness navigation paradigm to deal with flight deviation and 
design an angle robustness point-to-point navigation model for adaptively predicting direction angle.
In order to evaluate vision-based navigation methods in real scenarios, we collect a new dataset UAV_AR368 and 
design the Simulation Flight Testing Instrument (SFTI) using Google Earth, 
which can simulate real-world flight deviation effectively and avoid the cost of real flight testing.

# Project Structure             
│  bs_train.py  
│  bs_train_one.py  
│  datasets.py  
│  main.py  
│  model.py  
│  README.md  
│  requirements.txt  
│  utils.py  
│  
├─baseline  
│  │  bs_datasets.py  
│  │  bs_models.py  
│  
└─processOrder  
   │  process_datasets.py  
   │  
   ├─100  
   │  │  cluster_centre.txt  
   │  │  cluster_labels.txt  
   │  │  cluster_pics.txt  
   │  │  
   │  ├─all_class  
   │  │  
   │  └─cluster_labels  
   │  
   ├─datasets  
   │  
   └─order

# Install
`pip install -r requirements.txt`

# Prepare Datasets
Our dataset UAV_AR368 will be publicly available at xxxx
In this dataset, we can find a directory named 'order'.
In the directory, there are 368 subdirectories which represent realistic UAV flight routes.
In each route, there are different number of images equipped with specified coordinates from the start point to the end point.

To use this dataset to train / test our model, you should run the commands below:
`mkdir processOrder
mv order processOrder/
cd processOrder
python process_datasets.py`

# Train
`python main.py \
--dist-url 'tcp://localhost:10001' \
--multiprocessing-distributed --world-size 1 --rank 0`

# Test
please reference to project AngleRobustnessTest

# Citation
