# Projet Deep Learning II

## *Tristan François & Théo Morvan*

## Installation

Quelques librairies standard de Python sont nécessaires pour lancer le projet. Pour les installer, il suffit de lancer la commande suivante :

```bash
pip install numpy pandas matplotlib seaborn scipy
```

## Données

Les données doivent être placées dans le dossier `./data/` à la racine du projet. Les données de `Binary AlphaDigits` peuvent être téléchargées à l'adresse indiqué dans l'énoncé : [http://www.cs.nyu.edu/~roweis/data.html](http://www.cs.nyu.edu/~roweis/data.html). Le lien vers le dataset `MNIST` n'était par contre plus valide, nous avons donc récupéré le dataset sur [kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).

## Lancement

Nous avons fait tourner nos expériences dans deux notebooks Jupyter: `test.ipynb` et `study.ipynb`. Le notebook `test.ipynb` présente l'entraînement et la génération sur `Binary AlphaDigits`, tandis que nous avons utilisé `study.ipynb` pour générer les diagrammes utilisés dans l'étude sur les données de `MNIST`. Pour reproduire nos résultats, il suffit de faire tourner les deux notebooks.