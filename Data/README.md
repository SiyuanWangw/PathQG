**original** is the directory to store the original data SQuAD1.1 and Glove data.

**Preproces** includes scripts to preprocess the experimanetal data:
    * read_original_data.py: is to read the original SQuAD1.1 data following "Learning to Ask: Neural Question Generation for Reading Comprehension" ;
    * Graph_Structure.py: is the class of Graph, Node and Edge we construcetd;
    * SceneGraph_Constructor.py: is the script to automatically construct KG for each text using SceneGraph Parser;
    * EntityExtraction_Dandelion.py: is another method we tried to extract entities from texts to forming KG;
    * Preprocessing.py: is some preprocessing operations for data;
    * get_Glove_embedding.py: to initialize the word embedding using Glove;
    * utils: contains some auxiliary functions. 

**processed** contains the data we have processed using scripts in **Preprocess**.