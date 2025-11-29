# inspect.py — quick dataset sanity & label distribution plot
import os
from load_data import load_ciciot
from config import SEQUENCE_LENGTH, SMALL_SAMPLE, N_ROWS_SMALL
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




def inspect():
    df = load_ciciot()
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())


    if 'Label' in df.columns:
        df = df.rename(columns={'Label': 'label'})


    if 'label' not in df.columns:
        raise ValueError("CSV must contain 'label' column")


    dist = df['label'].value_counts()
    print('\nLabel Distribution:\n', dist)


    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(8,6))
    dist.plot(kind='bar')
    plt.title('Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/label_dist.png', dpi=200)
    plt.close()


    print('\nDefault SEQUENCE_LENGTH:', SEQUENCE_LENGTH)
    if SMALL_SAMPLE:
        print(f"NOTE: SMALL_SAMPLE enabled -> using {N_ROWS_SMALL} rows")


if __name__ == '__main__':
    inspect()