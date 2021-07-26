Automatic Ground Truth Generation in CARLA 
=============================================

This Python API gets a variety of data from CARLA Simulator and also helps automatically generate Ground Truth according to the Darknet YOLO Framework.Ground Truth on the objects provided are vehicle and pedestrian.

<img src="https://user-images.githubusercontent.com/36737287/82050024-5c1edd00-96f2-11ea-810d-3f8fcc664901.png" width="100%">

Installation
--------------
> cd [carla directory]/PythonAPI/examples  
> git clone https://github.com/JangJaeSung/carla_ground_truth.git

Extraction of CARLA Data
---------------------------
This API helps you information about the RGB Image, Semantic Segmentation Image and Bounding Box of Objects from the CARLA Server. In addition, it allows users to capture these informations at regular intervals and keyboard input.

* CARLA Server on  
> ./CarlaUE4.sh

* Spawn Object and Change Various Weathers (CARLA Python example). Link: [https://github.com/carla-simulator/carla/tree/master/PythonAPI/examples]  
Spawn NPC (Vehicles and Walkers).
> ./spawn.py -n *N* -w *W*  
> ./weather.py  

**Execute Generating Code**  
Obtain information about RGB Image, Semantic segmentation Image and Bounding Boxes for Ground Truth Generation from CARLA.
When a period (loop N) is specified, data capture is executed at that interval. This API helps users control the start and end of automatic capture through keyboard input. It also provides a feature that allows users to capture the scene they want immediately at the moment.  
> ./extract.py -l *N*   

These datas (RGB, Segmentation and Bounding Box) are stored in different folders at the same time.


Generation of Ground Truth for Darknet YOLO Framework
------------------------
This API helps to post-process the data obtained above and to create a Ground Truth for Darknet YOLO Framework.  
> ./generate.py  

Data for learning is stored in folder 'custom_data', and the results of the bounding box generated in folder 'draw_bounding_box' can be seen.
It will create .txt file for each .png rgb image file with the same name in the same directory. Ground truth formet of Darknet is shown below.
> [object's class] [x_center] [y_center] [image_width] [image_height]

* Object number
Vehicle : 0, Person (Pedestrian) : 1
* Other data
Datas without object's class number are expressed between 0 and 1.

* In addition, the image path to train is saved in 'my_data' folder in the form of a .txt file.
> custom_data/rgb1.png  
> custom_data/rgb2.png  
> custom_data/rgb3.png  
