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

```bash
mkdir data
wget "https://storage.googleapis.com/kaggle-data-sets/27352/34877/compressed/mnist_test.csv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230326%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230326T142708Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=49a1141b51f10569e0112ab7b7b2d550a8631a3de51019a2ebe25a4748a0266fb860347aa7937a1a5ebfcbc42c994b5626ee77419f6d4c6a18f2a847d2bd6d1988bb4d79f632a60df5efe9f0d1b6801a96b116ae825fe0fa90500ea18a61a0b8ea7e8b708c3b173f6acd3ccf2d0701775e0801a5ebde3b771d6ad6eaf9bd8750eba65e70f252a353812b254f462c20c73706a305f0af60e2cdee78389b19f34bde9d036401630d3b3e535ee4db900a22b6af0ccaa0d1c7527536676a0444ba312f087a729f8b22165cf8578f1a999efa2666c6f6145027fb11c31c4a2f22f83b894648dfd255f78da9a0ed85037974683cc9fccb80628fa0afc919517e044f0f" -O data/mnist_test.csv.zip
wget "https://storage.googleapis.com/kaggle-data-sets/27352/34877/compressed/mnist_train.csv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230326%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230326T142937Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=4b53a63b91f07cd4aba94d7c720cea72fc2cf8b9d3cb6536188ec8d4faf1c56577cd646c5856cf666338ae5ea4d015fff5080104bb4ac51a513c664b75b4dae7b72c80fa1d975ec7acf8a9e84142e8330f1efa4cc4421adff4c4975537245f0705249c6c40432011316fea2ec4a3793d9be7a4acc9205a133ada572584dd0944fdebaaaf3544126dd5044332f764fdcdd10159a5b612f8f64762df2af3f0aad0696a49a9045ce6ab3a3cd615adffc72e6e62ae6ea423b2998432146c4ada599f4f48bd4d3af81cab096aaa84dc75bcf290f2a746d0c8d608ca33c8d960d204fcf21ce9a275fd4ed01d2a70e62bddaad7d494cce6e43088521af36bb798a06759" -O data/mnist_train.csv.zip
unzip data/mnist_test.csv.zip -d data/
unzip data/mnist_train.csv.zip -d data/
```

## Lancement

Nous avons fait tourner nos expériences dans deux notebooks Jupyter: `test.ipynb` et `study.ipynb`. Le notebook `test.ipynb` présente l'entraînement et la génération sur `Binary AlphaDigits`, tandis que nous avons utilisé `study.ipynb` pour générer les diagrammes utilisés dans l'étude sur les données de `MNIST`. Pour reproduire nos résultats, il suffit de faire tourner les deux notebooks.

Enfin les tests concernant la partie bonus ont été effectués dans le notebook `test_vae.ipynb` et permettent de relancer l'entrainement des différents modèles ainsi que la génération d'images.