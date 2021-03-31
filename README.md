# relatable_clothing

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Contains the models and dataset from the Relatable Clothing papers.

To run, download download the Relatable Clothing model [here](https://drive.google.com/file/d/1IgtJKaokecn3OL6JR1Nel2KSnz18gZ_1/view?usp=sharing) and place .tar file model in

```
root
|
|--pretrained_model
   |
   |--optimal_model
      |
      |--optimal_model.tar
```

To run inference on an image:

```
python3 main.py --img-path path/to/image/
```

The results of inference are saved under

```
root
|
|--detections
```



## Datasets

The original Relatable Clothing dataset is found [here](https://drive.google.com/file/d/1mf4VXJw2Wbs0KV43u7NrbC3jbHSpgeXt/view?usp=sharing). Details on this dataset may be found in the original Relatable Clothing paper available at https://arxiv.org/abs/2007.10283.

The metadata for the segmentations subset of [Open Images V6](https://storage.googleapis.com/openimages/web/index.html) and [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) is found [here](https://drive.google.com/file/d/1TnLKwBPV9L7VoblolXasAjtX3oXt6wuj/view?usp=sharing). 

## Paper

Access to the previous Relatable Clothing paper remains available at https://arxiv.org/abs/2007.10283.

