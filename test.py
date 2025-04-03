import pandas as pd
import glob

csv_files = glob.glob("./*.csv")

dfs = [pd.read_csv(file) for file in csv_files]

merged_df = pd.concat(dfs, axis=1)

merged_df.to_csv("merged_output.csv", index=False)

print("合并完成，已保存为 merged_output.csv")
