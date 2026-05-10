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

SEEDS = [4567, 1234, 9999]  # Uma seed diferente para cada execução

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir, "spambase_treinamento.csv")
    test_path = os.path.join(current_dir, "spambase_teste.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        train_path = "spambase_treinamento.csv"
        test_path = "spambase_teste.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df

def shuffle_and_split(train_df, test_df, seed):
    # Junta tudo, embaralha com a seed da execução e divide novamente
    full_df = pd.concat([train_df, test_df]).reset_index(drop=True)
    shuffled_index = np.random.RandomState(seed).permutation(full_df.index)
    full_df = full_df.loc[shuffled_index].reset_index(drop=True)

    train_count = int(len(full_df) * 0.8)
    df_train = full_df.iloc[:train_count]
    df_test = full_df.iloc[train_count:]

    X_train = df_train.drop(columns=["is_spam"])
    y_train = df_train["is_spam"]
    X_test = df_test.drop(columns=["is_spam"])
    y_test = df_test["is_spam"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test

def evaluate_classifiers():
    try:
        train_df, test_df = load_data()
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

    for i, seed in enumerate(SEEDS):
        X_train, y_train, X_test, y_test = shuffle_and_split(train_df, test_df, seed)
        print(f"Execução {i+1} (seed={seed}):")
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