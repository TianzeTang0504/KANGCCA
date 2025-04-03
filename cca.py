import torch
import pandas as pd
from linear_gcca import linear_gcca

# === 加载四组视图 ===
protein = torch.tensor(pd.read_csv("exp.csv").values, dtype=torch.float64)
clinical = torch.tensor(pd.read_csv("clinical.csv").values, dtype=torch.float64)
dna = torch.tensor(pd.read_csv("meth.csv").values, dtype=torch.float64)
rna = torch.tensor(pd.read_csv("miRNA.csv").values, dtype=torch.float64)

views = [dna, rna, protein, clinical]

# === 运行 Linear GCCA ===
outdim_size = 64
lgcca = linear_gcca()
lgcca.fit(views, outdim_size)
outputs = lgcca.test(views)

# === 取平均输出并保存 ===
z = sum(outputs) / len(outputs)  # 可替换为 torch.cat(outputs, dim=1)
df_z = pd.DataFrame(z.cpu().numpy())
df_z.to_csv("onlycca.csv", index=False)
