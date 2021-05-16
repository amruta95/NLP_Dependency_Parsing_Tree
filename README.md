# NLP_Dependency_Parsing_Tree

In this project, I have implemented Dependency Parsing using two methods:

Arc-Standard Algorithm (From Incrementality in Deterministic Dependency Parsing(2004, Nivre))

Neural Networks (implemented feature extraction, neural network architecture including activation function, loss function) (From A Fast and Accurate Dependency Parser using Neural Networks(2014, Danqi and Manning))

Files in this repository:

- Dependency_Parser.py: This file is the main script for training dependency parser.

- DependencyTree.py The dependency tree class file.

- Parsing_System.py This file contains the class for a transition-based parsing framework for dependency parsing.

- Configuration.py The configuration class file.

- Config.py This file contains all hyper parameters.

- Util.py This file contains functions for reading and writing CONLL file.

- data/ train.conll - train set, labeled dev.conll - dev set, labeled test.conll - test set, unlabeled