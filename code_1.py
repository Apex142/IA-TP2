import os
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Construire un échantillon
def buildSampleFromPath(path1, path2):
    sample = []
    for path, label in [(path1, 1), (path2, -1)]:
        for file in os.listdir(path):
            sample.append({'name_path': os.path.join(path, file), 'y_true_class': label})
    return sample

# Redimensionner l'image
def resizeImage(i, h, l):
    return i.resize((l, h))

# Calculer l'histogramme de l'image
def computeHisto(i):
    return np.array(i.convert('RGBA').histogram()) # Converti en RGBA pour les images en PNG car elles ont 4 canaux

# Mettre à jour resized_image et X_histo de l'entrée (image)
def updateImageEntry(entry, h, l):
    img = Image.open(entry['name_path'])
    entry['resized_image'] = resizeImage(img, h, l)
    entry['X_histo'] = computeHisto(entry['resized_image'])
    return entry

# Entraîner le modèle à partir des histogrammes
def fitFromHisto(S, algo):
    if (algo['name'] != 'NaiveBayes'):
        raise ValueError("Unknown algorithm")
    model = GaussianNB(**algo.get('hyper_param', {}))
    X = [entry['X_histo'] for entry in S]
    y = [entry['y_true_class'] for entry in S]
    model.fit(X, y)
    return model

# Prédire les classes à partir des histogrammes
def predictFromHisto(S, model):
    X = [entry['X_histo'] for entry in S]
    y_pred = model.predict(X)
    for i in range(len(S)):
        S[i]['y_predicted_class'] = y_pred[i]
    return y_pred

# Calculer l'erreur empirique
def computeEmpiricalError(S):
    erreurs = 0  
    for entry in S:
        if entry["y_true_class"] != entry["y_predicted_class"]:  
            erreurs += 1  
    erreur_empirique = erreurs / len(S) 
    return erreur_empirique

# Calculer l'erreur de validation avec la méthode hold-out
def holdOutError(S, algo, test_size=0.2):
    X = [entry['X_histo'] for entry in S]
    y = [entry['y_true_class'] for entry in S]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    if algo['name'] != 'NaiveBayes':
        raise ValueError("Unknown algorithm")
    
    model = GaussianNB(**algo.get('hyper_param', {}))
    model.fit(X_train, y_train)
    
    return accuracy_score(y_test, model.predict(X_test))

if __name__ == "__main__":
    # Chemins vers les dossiers contenant les images
    path_mer = "Data/Mer"
    path_ailleurs = "Data/Ailleurs" 

    # Construire l'échantillon à partir des chemins
    S = buildSampleFromPath(path_mer, path_ailleurs)

    # Mettre à jour chaque entrée de l'échantillon avec l'image redimensionnée et son histogramme
    for i, entry in enumerate(S):
        S[i] = updateImageEntry(entry, 128, 128)

    # Définir l'algorithme à utiliser
    algo = {
        'name': 'NaiveBayes',
        'hyper_param': {} 
    }

    # Entraîner le modèle à partir des histogrammes
    model = fitFromHisto(S, algo)

    # Prédire les classes à partir des histogrammes
    y_pred = predictFromHisto(S, model)

    # Afficher les résultats pour les premières images
    for i in range(min(5, len(S))):
        print(f"Image : {S[i]['name_path']}")
        print(f"   - Classe réelle : {S[i]['y_true_class']}")
        print(f"   - Classe prédite : {S[i]['y_predicted_class']}")

    # Calculer et afficher l'erreur empirique
    nb_error = computeEmpiricalError(S)
    print(f"Erreur empirique : {nb_error}")

    # Calculer et afficher l'erreur de validation avec la méthode hold-out
    hold_out_error = holdOutError(S, algo)
    print(f"Erreur de validation : {hold_out_error}")