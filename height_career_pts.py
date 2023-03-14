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
merged_salaries = merged.groupby('_id')['salary'].mean().reset_index()

# height vs average points per game
fig = px.scatter(merged, x='height', y='career_PTS', color='name',
                 labels={'height': 'Height', 'career_PTS': 'Career_PTS'})
fig.update_layout(title='Career_PTS vs Salary', showlegend=False)
fig.write_image('Height vs Career_PTS.png')

# Fit a linear regression model
X = merged[['height']]
y = merged['career_PTS']
model = LinearRegression()
model.fit(X, y)

# Evaluate the model's performance
score = model.score(X, y)
print('R^2 Score:', score)
