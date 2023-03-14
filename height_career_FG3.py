import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

data_a = pd.read_csv('players.csv')
data_b = pd.read_csv('salaries_1985to2018.csv')

# Function
def _converting_machine(input_data):
    feet, inches = input_data.split("-")
    return int(feet) * 12 + int(inches)

# Datasets
merged = data_a.merge(data_b, left_on="_id", right_on="player_id")
merged['height'] = merged['height'].apply(_converting_machine)
merged = merged[merged['career_FG3%'] != '-']
merged_salaries = merged.groupby('_id')['salary'].mean().reset_index()


# height vs 3-point field goal percentage
merged = merged[merged['career_FG3%'] != '-']
merged['career_FG3%'] = pd.to_numeric(merged['career_FG3%'])
merged = merged.rename(columns={'career_FG3%': 'career_FG3'})
fig = px.scatter(merged, x='height', y='career_FG3', color='name',
                 labels={'height': 'Height', 'career_FG3': 'Career_FG3'})
fig.update_layout(title='Height vs Career_FG3', showlegend=False)
fig.write_image('Height vs Career_FG3.png')


# Fit a linear regression model
X = merged[['height']]
y = merged['career_FG3']
model = LinearRegression()
model.fit(X, y)

# Evaluate the model's performance
score = model.score(X, y)
print('R^2 Score:', score)

# filter outliers using the career_FG3 variable
q1 = merged['career_FG3'].quantile(0.25)
q3 = merged['career_FG3'].quantile(0.75)
iqr = q3 - q1
career_FG3_range = (q1 - 1.5*iqr, q3 + 1.5*iqr)
merged_filtered = merged.query('career_FG3 >= @career_FG3_range[0] and career_FG3 <= @career_FG3_range[1]')

# create a scatter plot with filtered data
fig = px.scatter(merged_filtered, x='height', y='career_FG3', color='name',
                 labels={'height': 'Height', 'career_FG3': 'Career_FG3'})

# customize the plot title and hide the legend
fig.update_layout(title='Height vs Career_FG3 (excluding outliers)', showlegend=False)

# save the plot as an image
fig.write_image('Height vs Career_FG3 (filtered).png')

# Fit a linear regression model
X = merged_filtered[['height']]
y = merged_filtered['career_FG3']
model = LinearRegression()
model.fit(X, y)

# Evaluate the model's performance
score = model.score(X, y)
print('R^2 Score:', score)
