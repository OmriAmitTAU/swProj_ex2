import pandas as pd

vec_num = [-1]
vec_size = [-1]

def init_vector_list(input_data):
    vectors = []
    for _, row in input_data.iterrows():
        vector = tuple(float(point) for point in row.values)
        vectors.append(vector)
    return vectors

def init_vector_list(vec_df):
    vec_df = vec_df.set_index(vec_df.columns[0])
    vec_list = vec_df.values
    vec_size[0] = vec_list.shape[1]
    return vec_list

df1 = pd.read_csv("./tests/input_1_db_1.txt")
df2 = pd.read_csv("./tests/input_1_db_2.txt")
merged_df = pd.merge(df1, df2, left_on=df1.columns[0], right_on=df2.columns[0], how="inner").drop(df2.columns[0], axis=1)

def init_vector_list(vec_df):
    vec_df = vec_df.set_index(vec_df.columns[0])
    vec_list = vec_df.values
    n = vec_list.shape[0]
    vec_size[0] = vec_list.shape[1]
    return vec_list

d = merged_df.iloc[0].notnull().sum() -1
print(d)
print(len(merged_df))
print(init_vector_list(merged_df))

