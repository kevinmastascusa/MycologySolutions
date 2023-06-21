import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

# Load the data
df = pd.read_csv('mushrooms.csv')

# Convert categorical variables into integers
le = LabelEncoder()
df = df.apply(le.fit_transform)

# Separate the data into features (X) and target (y)
X = df.drop('class', axis=1)  # Assuming 'class' is the column with information about edibility
y = df['class']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create a decision tree classifier and fit it to our data
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# Test the model
accuracy = clf.score(X_test, y_test)
print(f'Model accuracy: {accuracy}')
