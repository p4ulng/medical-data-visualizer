import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = np.where(
    (df['weight']/((df['height']/100)**2)) >25,
    1,
    0
)
df['overweight']=df['overweight'].astype(int)
# 3
df['cholesterol']=np.where(
    df['cholesterol']== 1,
    0,
    1
)
df['gluc']=np.where(
    df['gluc']== 1,
    0,
    1
)
df[['gluc','cholesterol']]=df[['gluc','cholesterol']].astype(int)
# print(df)
# print("hello")
# df_cat = df[['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']]
# df_cat = pd.melt(
#     df,
#     id_vars=['cardio'],
#     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
# )
# print("\nMelted DataFrame Structure:")
# print(df_cat.head(10))  # show first 10 rows to better understand the transformation
# print("\nShape:", df_cat.shape)  # show resulting dimensions

# df_cat=df_cat.groupby(['cardio']).value_counts().rename('count').reset_index()
# df_cat['count']=df_cat['count'].astype('int64')
# print(df_cat)

# fig = sns.catplot(
#     data=df_cat,
#     x='variable',
#     y='count',
#     hue='value',
#     col='cardio',
#     kind='bar'
# )

# plt.show()  # This will display in VS Code's plot viewer

def draw_cat_plot():
    # Melt the data
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    )

    # Group by category and value
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # Create the catplot (returns a FacetGrid)
    fig= sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        kind='bar',
        col='cardio',
        hue='value'
    ).set_xlabels("variable").set_ylabels("total")


    # Do not modify the next two lines
    fig = fig.fig  # this is the actual matplotlib.figure.Figure
    fig.savefig('catplot.png')
    return fig
# 10
def draw_heat_map():
    # 11
    df_heat=df[(df['ap_lo'] <= df['ap_hi']) & 
           (df['height'] >= df['height'].quantile(0.025)) &
           (df['height'] <= df['height'].quantile(0.975)) &
           (df['weight'] >= df['weight'].quantile(0.025)) &
           (df['weight'] <= df['weight'].quantile(0.975))
           ]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr))


    # 14
    fig, ax = plt.subplots()

    # 15
  
    sns.heatmap(
        corr, 
        mask=mask,
        annot=True,      # Show correlation values
        fmt='.1f',       # Format to 1 decimal place
        cmap='coolwarm', # Color scheme
        square=True      # Make cells square
    )
    plt.title('Correlation Matrix Heatmap')


    # 16
    fig.savefig('heatmap.png')
    return fig
