# Sketched-Image-Segmentation

Sketched Image Segmentation 

My Internship works at Los Alamos National Lab

Model: 
Point-Shooting Method
U-Net, HR-Net, Transformer

Deep Learning Framework: Pytorch, Keras 

Instructions: 


1. Point shooting: To get the boudng box along subimages of objects please run point-shooting-method-sketched-images.py 
   
   Command: python point-shooting-method-sketched-images.py
   
   To get the boudng box along subimages of objects and coordinates please run Point-shooting-bounding-box-roi-coordinates.py
   
   To get the bouding box from the contour, please run Point-shooting-Method-sketced-image-contour-extractino.py
   
2. To train U-Net: Please run U-Net-Segmentaion.py and for HR-Net please run  HR-Net.py

   command: python HR-Net.py 
   
   For testing, please run UNet-HR-Net-testing.py
   
3. Please download Medical Transformer (https://github.com/jeya-maria-jose/Medical-Transformer) and put testing_transformer.py, test_ex.py,testing_transformer_ex.py files on that folder. Trained model work on sketeched image segmentation.
 
 To train tarnsformer, please run the following command:

   python train.py --train_dataset "enter train directory" --val_dataset "enter validation directory" --direc 'path for results to be saved' --batch_size 4 --epoch 400 --save_freq 10 --modelname "gatedaxialunet" --learning_rate 0.001 --imgsize 128 --gray "no"

   For testing: 
   1.  To generate and resize ground-truth image using point-shooting and also resize the orginal images: 
   
   please run testing_transformer.py
   
   2. To only resize the input images and then apply it to the transformer model: 
   
   please run testing_transformer_ex.py
