# PathQG
The source code of Paper "PathQG: Neural Question Generation from Facts".

## Requires python 3 and Pytorch
```
pip install -r requirements.txt
```

## Data Preprocessing

We provide our split and preprocessed data and constructed Knowledge Graphs in the **processed** directory. You can **directly utilize it and skip the following operations**. 
Or you can also start from the original data and preprocess it by the following steps. 
The original SQuAD1.1 data is downloaded from xinyadu ("Learning to Ask: Neural Question Generation for
Reading Comprehension")'s github: https://github.com/xinyadu/nqg/tree/master/data/processed.
```
cd ../Preprocess
python read_original_data.py
python SceneGraph_Constructor.py
python Preprocessing.py
```

Then we need to download the word embeddings and load the data for our models.

```
cd original 
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip 

cd ../../Models/Graph_Models
python SceneGraph_LoadData.py
python ../../Data/Preprocess/get_Glove_embedding.py
```

## Usage
To train and evaluate a model, 
```
cd Models/Graph_Models
python *_Solver.py
```
Here * means that corresponding model, for example, **PathQG_Solver.py** is the trainer and tester for PathQG model.

## Code structure
* **Data** contains all original and processed data, with scripts preprocessing them;
* **SceneGraphParser** is an open python implementation of scenegraph parser online from https://github.com/vacancy/SceneGraphParser; 
* **Models** is composed of different models and their solver (trainer and test);
* **Model_Data** saves the trained model files; 
* **Evaluator** contains scripts to calculate the evaluation metrics;

## Constructed KG
The constructed Knowledge Graphs by SceneGraph Parser is provided in the directory **Data/processed/SQuAD1.0/Graph_Analysis/#**, here # means **train**, **val** or **test**.