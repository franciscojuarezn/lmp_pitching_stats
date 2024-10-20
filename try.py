import pandas as pd
import os


df = pd.read_csv('stats_data/players_data_2024.csv', encoding='utf-8')

print(df.POS.unique())

print(df[df['POS'].isna()])

df_std = pd.read_csv('stats_data/df_standard_stats_2024 (8).csv')
df_adv = pd.read_csv('stats_data/df_advanced_stats_2024 (9).csv')

print(df_std.head())

print(df_std.columns.values)
print(df_adv.columns.values)

print(df_adv['LD%'].head(10))

