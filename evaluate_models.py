import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings("ignore")

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir, "spambase_treinamento.csv")
    test_path = os.path.join(current_dir, "spambase_teste.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        train_path = "spambase_treinamento.csv"
        test_path = "spambase_teste.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=["is_spam"])
    y_train = train_df["is_spam"]
    X_test = test_df.drop(columns=["is_spam"])
    y_test = test_df["is_spam"]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, y_train, X_test_scaled, y_test

def evaluate_classifiers():
    try:
        X_train, y_train, X_test, y_test = load_data()
    except Exception:
        print("\nERRO: Arquivos .csv não encontrados na pasta do script.")
        return

    classifiers = {
        "Perceptron": Perceptron(),
        "Bayes": GaussianNB(),
        "MLP": MLPClassifier(max_iter=500),
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier()
    }
    
    results = {name: [] for name in classifiers.keys()}
    
    print("\nIniciando avaliação dos classificadores (3 execuções)...")
    print("-" * 50)
    
    for i in range(3):
        print(f"Execução {i+1}:")
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name].append(acc)
            print(f"  - {name}: {acc:.4f}")
        print("-" * 30)
        
    print("\nRESULTADOS FINAIS (Média de Acurácia):")
    print("-" * 50)
    final_summary = []
    for name, accs in results.items():
        mean_acc = np.mean(accs)
        print(f"{name:25} | Média: {mean_acc:.4f}")
        final_summary.append({"Classificador": name, "Média Acurácia": f"{mean_acc:.4f}"})
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_acuracia.csv")
    pd.DataFrame(final_summary).to_csv(output_path, index=False)
    print(f"\nResumo salvo em: {output_path}")

if __name__ == "__main__":
    evaluate_classifiers()
