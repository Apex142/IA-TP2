# Rapport TP2 - Classification d'images mer/non-mer

**Étudiant:**
rouabhi nihal nour elhouda,
sbai eya,
MARTIN Mehdi

**Date:** 03 février 2025

## 1. Objectif du TP

L'objectif de ce TP était d'implémenter un système de classification binaire d'images permettant de distinguer les paysages maritimes des autres paysages. Au début du TP, nous avons trié les images présentes dans les dossiers `/data/mer` et `/data/ailleurs` car certaines images n'étaient pas à leur place et pouvaient affecter les données et fausser les résultats.

Nous avons effectué les modifications suivantes :

- Supprimé `mlxat` et `xlou` de `AILLEURS` car ce sont des images de `MER`
- Supprimé `naqq` et `tagtoo` de `MER` car ce sont des images de `AILLEURS`
- Déplacé `oag521` de `MER` à `AILLEURS`
- Déplacé `pillq` de `MER` à `AILLEURS`

## 2. Implémentation

### 2.2 Fonctions principales implémentées

1. `buildSampleFromPath(path1, path2)`:

- Construit l'échantillon à partir des répertoires d'images
- Retourne une liste de dictionnaires contenant les chemins et labels

2. `resizeImage(i, h, l)`:

- Redimensionne les images à une taille uniforme

3. `computeHisto(i)`:

- Calcule l'histogramme des couleurs
- Convertit l'image en RGBA pour gérer les PNG
- Retourne un vecteur numpy de 1024 valeurs (256\*4 canaux) car les images sont en JPEG (3 canaux) et d'autres en PNG (4 canaux), nous avons choisi de convertir les images en RGBA (4 canaux) car cela permet de traiter uniformément toutes les images indépendamment de leur format.

4. `predictFromHisto(S, model)`:

- Prédit les classes des images
- Met à jour les prédictions dans l'échantillon

### 2.3 Évaluation

Nous avons implémenté

1. `computeEmpiricalError()`: Calcule l'erreur empirique sur l'ensemble d'apprentissage
2. `holdOutError()`: Estime l'erreur réelle par validation croisée (hold-out)

## 3. Résultats

- Structure de données en utilisant les dictionnaires et les listes
- Traitement d'images standardisé (redimensionnement uniforme)
- Classification basée sur les histogrammes en RGBA
- Évaluation avec deux métriques différentes
