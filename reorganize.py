import pandas as pd
import numpy as np

# read and shuffle the data set
Eye_dataframe = pd.read_csv("./complete.csv",index_col=0)
Eye_dataframe = Eye_dataframe.sample(frac=1).reset_index(drop=True)

# reorganize the dataset
rows = []
person = 1
# for each row in the original dataset
#   take out right eye metrics and make a row
#   take out left eye metrics and make a row
#   go to next person
for k,v in Eye_dataframe.iterrows():
    rows.append([person, "right", v['Age'], v['Sex'], v['MRD-1(R)'], v['PTB (R)'], v['TPS (R)'],v['Ethnicity']])
    rows.append([person, "left", v['Age'], v['Sex'], v['MRD-1 (L)'], v['PTB (L)'], v['TPS (L)'],v['Ethnicity']])
    person += 1

# make the rows into a dataframe and set column names
random_effect_data = pd.DataFrame(rows, columns=["Person","Eye","Age","Sex","MRD","PTB","TPS","Ethnicity"])
# output the reoganized dataset
random_effect_data.to_csv("./random_effect_data.csv")
