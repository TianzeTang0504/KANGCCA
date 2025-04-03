import numpy as np
import pandas as pd

input_file = 'data/TCGA_BRCA/miRNA664.txt'
output_file = 'miRNA.csv'

data = np.loadtxt(input_file, delimiter=None)

transposed_data = data.T

df = pd.DataFrame(transposed_data)

df.to_csv(output_file, index=False)

print(f"转置后的数据已保存为 {output_file}")
