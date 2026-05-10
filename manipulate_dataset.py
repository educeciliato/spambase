import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype
from pathlib import Path
import json

BASE_DIR = Path(__file__).parent

def load_json(file_path: Path):
    if not file_path.exists():
        return {}
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)

def save_json(file_path: Path, data):
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

def apply_configured_column_mappings(df: pd.DataFrame, config: dict):
    map_columns = config.get("map-columns", {})

    for column_name, mapping_name in map_columns.items():
        if column_name not in df.columns:
            continue

        mapping_path = BASE_DIR / f"{mapping_name}.json"
        if not mapping_path.exists():
            continue

        ranges = load_json(mapping_path)
        ordered_ranges = list(ranges.items())

        def map_value(value):
            numeric_value = pd.to_numeric(value, errors="coerce")
            if pd.isna(numeric_value):
                return value
            for label, upper_bound in ordered_ranges:
                if numeric_value <= upper_bound:
                    return label
            return ordered_ranges[-1][0]

        df[column_name] = df[column_name].map(map_value)

    return df

def has_missing_markers(value) -> bool:
    value_str = str(value).strip()
    return pd.isna(value) or value_str in {"?", "-1"}

def remove_missing_values(df: pd.DataFrame):
    # Otimização: usar dropna diretamente se possível ou boolean indexing
    mask = df.apply(lambda row: any(has_missing_markers(val) for val in row), axis=1)
    df = df[~mask]
    return df.dropna()

def transform_string_columns(df: pd.DataFrame):
    mappings = {}
    for coluna in df.columns:
        if is_string_dtype(df[coluna]):
            classes = df[coluna].unique()
            mapping = {str(label): int(idx) for idx, label in enumerate(classes)}
            df[coluna] = df[coluna].map(mapping)
            mappings[coluna] = mapping
    return df, mappings


def split_dataset(df: pd.DataFrame, train_size: float = 0.8):
    train_count = int(len(df) * train_size)
    df_train = df.iloc[:train_count].reset_index(drop=True)
    df_test = df.iloc[train_count:].reset_index(drop=True)
    return df_train, df_test

def get_column_names():
    # Nomes baseados no arquivo spambase.names
    cols = [
        "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", "word_freq_our",
        "word_freq_over", "word_freq_remove", "word_freq_internet", "word_freq_order", "word_freq_mail",
        "word_freq_receive", "word_freq_will", "word_freq_people", "word_freq_report", "word_freq_addresses",
        "word_freq_free", "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
        "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp",
        "word_freq_hpl", "word_freq_george", "word_freq_650", "word_freq_lab", "word_freq_labs",
        "word_freq_telnet", "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_85",
        "word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
        "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project", "word_freq_re",
        "word_freq_edu", "word_freq_table", "word_freq_conference", "char_freq_;", "char_freq_(",
        "char_freq_[", "char_freq_!", "char_freq_$", "char_freq_#", "capital_run_length_average",
        "capital_run_length_longest", "capital_run_length_total", "is_spam"
    ]
    return cols

def process_dataset():
    config_path = BASE_DIR / "config.json"
    config = load_json(config_path)
    
    data_path = BASE_DIR / "spambase.data"
    if not data_path.exists():
        print(f"Erro: Arquivo {data_path} não encontrado.")
        return

    # Carregar o dataset definindo os nomes das colunas
    df = pd.read_csv(data_path, header=None, names=get_column_names())
    print(f"Dataset carregado com {len(df)} linhas.")

    df = remove_missing_values(df)
    print(f"Linhas após remover valores ausentes: {len(df)}")

    df = apply_configured_column_mappings(df, config)
    
    df_numeric, mappings = transform_string_columns(df)
    
    df_train, df_test = split_dataset(df_numeric, train_size=0.8)

    output_train_path = BASE_DIR / "spambase_treinamento.csv"
    df_train.to_csv(output_train_path, index=False)
    
    output_test_path = BASE_DIR / "spambase_teste.csv"
    df_test.to_csv(output_test_path, index=False)

    groupings_path = BASE_DIR / "mappings.json"
    save_json(groupings_path, mappings)

    print(f"Arquivo de treinamento criado em: {output_train_path}")
    print(f"Arquivo de teste criado em: {output_test_path}")
    print(f"Agrupamentos salvos em: {groupings_path}")

def main():
    process_dataset()

if __name__ == "__main__":
    main()
