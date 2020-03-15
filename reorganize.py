import pandas as pd
import numpy as np


Eye_dataframe = pd.read_csv("./complete.csv",index_col=0)
Eye_dataframe = Eye_dataframe.sample(frac=1).reset_index(drop=True)

rows = []
person = 1
for k,v in Eye_dataframe.iterrows():
    rows.append([person, "right", v['Age'], v['Sex'], v['MRD-1(R)'], v['PTB (R)'], v['TPS (R)'],v['Ethnicity']])
    rows.append([person, "left", v['Age'], v['Sex'], v['MRD-1 (L)'], v['PTB (L)'], v['TPS (L)'],v['Ethnicity']])
    person += 1

random_effect_data = pd.DataFrame(rows, columns=["Person","Eye","Age","Sex","MRD","PTB","TPS","Ethnicity"])

random_effect_data.head()

random_effect_data.to_csv("./random_effect_data.csv")
