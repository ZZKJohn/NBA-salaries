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

# height vs average salary
fig = px.scatter(merged, x='height', y='salary', color='name',
                 labels={'Height': 'height', 'salary': 'Salary'})
fig.update_layout(title='Height vs Salary', showlegend=False)
fig.write_image('Height vs Salary.png')

# Fit a linear regression model
X = merged[['height']]
y = merged['salary']
model = LinearRegression()
model.fit(X, y)

# Evaluate the model's performance
score = model.score(X, y)
print('R^2 Score:', score)

# Calculate the IQR for the salary column
Q1 = merged['salary'].quantile(0.25)
Q3 = merged['salary'].quantile(0.75)
IQR = Q3 - Q1

# Filter the dataset to remove values outside of 1.5 times the IQR
merged_filtered = merged[(merged['salary'] >= Q1 - 1.5*IQR) & (merged['salary'] <= Q3 + 1.5*IQR)]

# Create the scatter plot with the filtered dataset
fig = px.scatter(merged_filtered, x='height', y='salary', color='name',
                 labels={'Height': 'height', 'salary': 'Salary'})
fig.update_layout(title='Height vs Salary (Filtered)', showlegend=False)
fig.write_image('Height vs Salary (Filtered).png')

# Fit a linear regression model
X = merged_filtered[['height']]
y = merged_filtered['salary']
model = LinearRegression()
model.fit(X, y)
score = model.score(X, y)
print('R^2 Score:', score)
