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

# Ignorar avisos de convergência para o MLP se necessário
warnings.filterwarnings("ignore")

def load_data():
    # Procura os arquivos no mesmo diretório do script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    train_path = os.path.join(current_dir, "spambase_treinamento.csv")
    test_path = os.path.join(current_dir, "spambase_teste.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        # Tenta procurar no diretório atual de trabalho caso o acima falhe
        train_path = "spambase_treinamento.csv"
        test_path = "spambase_teste.csv"

    print(f"Carregando dados de: {train_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=["is_spam"])
    y_train = train_df["is_spam"]
    X_test = test_df.drop(columns=["is_spam"])
    y_test = test_df["is_spam"]
    
    # Normalização é importante para Perceptron, MLP e SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, y_train, X_test_scaled, y_test

def evaluate_classifiers():
    try:
        X_train, y_train, X_test, y_test = load_data()
    except FileNotFoundError as e:
        print(f"\nERRO: Arquivos de dados não encontrados!")
        print("Certifique-se de que 'spambase_treinamento.csv' e 'spambase_teste.csv' estão na mesma pasta que este script.")
        return

    classifiers = {
        "Perceptron": Perceptron(),
        "Bayes (GaussianNB)": GaussianNB(),
        "MLP (Neural Network)": MLPClassifier(max_iter=500),
        "SVM (SVC)": SVC(),
        "Decision Tree": DecisionTreeClassifier()
    }
    
    results = {name: [] for name in classifiers.keys()}
    
    print("\nIniciando avaliação dos classificadores (3 execuções cada)...")
    print("-" * 50)
    
    for i in range(3):
        print(f"Execução {i+1}:")
        for name, clf in classifiers.items():
            # Treinar o modelo
            clf.fit(X_train, y_train)
            # Predizer
            y_pred = clf.predict(X_test)
            # Calcular acurácia
            acc = accuracy_score(y_test, y_pred)
            results[name].append(acc)
            print(f"  - {name}: {acc:.4f}")
        print("-" * 30)
        
    print("\nRESULTADOS FINAIS (Média de Acurácia):")
    print("-" * 50)
    final_summary = []
    for name, accs in results.items():
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"{name:25} | Média: {mean_acc:.4f} | Desvio: {std_acc:.4f}")
        final_summary.append({"Classificador": name, "Média Acurácia": f"{mean_acc:.4f}"})
    
    # Salvar resumo em CSV
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_acuracia.csv")
    pd.DataFrame(final_summary).to_csv(output_path, index=False)
    print(f"\nResumo salvo em: {output_path}")

if __name__ == "__main__":
    evaluate_classifiers()
