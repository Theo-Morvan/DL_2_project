# Projet Deep Learning II

## *Tristan François & Théo Morvan*

## Installation

Quelques librairies standard de Python sont nécessaires pour lancer le projet. Pour les installer, il suffit de lancer la commande suivante :

```bash
pip install numpy pandas matplotlib seaborn scipy
```

Concernant la partie Bonus sur les VAE, l'implémentation a été faite avec pytorch. Pour pouvoir lancer le notebook utilisé pour la visualisation, il suffit de lancer la commander suivante en supplément de la précédente pour windows

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

Tandis que pour mac la commande est la suivante (version sans GPU)

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Et pour Linux

```bash
pip3 install torch torchvision torchaudio
```

## Données

Les données doivent être placées dans le dossier `./data/` à la racine du projet. Les données de `Binary AlphaDigits` peuvent être téléchargées à l'adresse indiqué dans l'énoncé : [http://www.cs.nyu.edu/~roweis/data.html](http://www.cs.nyu.edu/~roweis/data.html). Le lien vers le dataset `MNIST` n'était par contre plus valide, nous avons donc récupéré le dataset sur [kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).

## Lancement

Nous avons fait tourner nos expériences dans deux notebooks Jupyter: `test.ipynb` et `study.ipynb`. Le notebook `test.ipynb` présente l'entraînement et la génération sur `Binary AlphaDigits`, tandis que nous avons utilisé `study.ipynb` pour générer les diagrammes utilisés dans l'étude sur les données de `MNIST`. Pour reproduire nos résultats, il suffit de faire tourner les deux notebooks.

Enfin les tests concernant la partie bonus ont été effectués dans le notebook `test_vae.ipynb` et permettent de relancer l'entrainement des différents modèles ainsi que la génération d'images.