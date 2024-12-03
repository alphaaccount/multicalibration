import pandas as pd

df1 = pd.read_csv('test_pos_final.txt', sep=',', header=None)
df2 = pd.read_csv('test_neg_final.txt', sep=',', header=None)

print("Previous number of positive pairs", len(df1))
X = pd.read_csv("../../../../UGE-Unbiased-Graph-Embedding/processed_data/deleted_nodes.csv")
value_set = set(X['Index'])


#for i in range(len(df1)):
#  if (df1.iloc[i,0] in nodes_set) or (df1.iloc[i,1] in nodes_set):
    

df1 = df1[~(df1.iloc[:, 0].isin(value_set) | df1.iloc[:, 1].isin(value_set))]
df2 = df2[~(df2.iloc[:, 0].isin(value_set) | df2.iloc[:, 1].isin(value_set))]


df1.to_csv('test_pos_final2.txt', index=False, header=False)
df2.to_csv('test_neg_final2.txt', index=False, header=False)

print("New number of positive pairs", len(df1))
