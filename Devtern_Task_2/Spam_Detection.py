# Email Spam Detection Using Machine Learning

# 1. Data Collection
import pandas as pd

# Loading the dataset from a local file (update the file path as necessary)
data = pd.read_csv('D:\DevTern_Task_2\spam_dataset1.csv')

# Display the first 3 rows
print(data.head(3))

# Display the last 3 rows
print(data.tail(3))

# 2. Data Organization
print(data.shape)  # gives information about the data i.e, total number of rows and columns
print(data.info())

# From the above info, it shows that there are no null values present
print(data.describe())  # gives the summary of the data

# 3. Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Renaming columns to match the expected format for consistency
data.rename(columns={'label': 'Category', 'text': 'Message'}, inplace=True)

# Countplot for 'Category' column
sns.countplot(data=data, x='Category')
plt.show()

category_counts = data.groupby('Category')['Category'].count()
print(category_counts)

# Total number of ham and spam mails
total_mails = len(data)
ham_percentage = (category_counts['ham'] / total_mails) * 100
spam_percentage = (category_counts['spam'] / total_mails) * 100

print('Ham :-', ham_percentage)
print('Spam :-', spam_percentage)

# Pie chart
plt.pie([ham_percentage, spam_percentage], labels=['Ham', 'Spam'], autopct='%1.1f%%')
plt.title('Mails by Category')
plt.show()

# 4. Data Preparation
# As the category data which is needed for detection has only string values, replacement by numbers is to be done.
data['Category'] = data['Category'].replace({'ham': 0, 'spam': 1})

# Dividing the dataset into input and output columns
x = data['Message']
y = data['Category']

# Importing needed libraries
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Feature extraction using TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_train_new = vectorizer.fit_transform(x_train)
x_test_new = vectorizer.transform(x_test)

# 5. Model Building
# (1) Logistic Regression
from sklearn.linear_model import LogisticRegression
model_1 = LogisticRegression()
model_1.fit(x_train_new, y_train)

# Model evaluation for Logistic Regression
score_1 = model_1.score(x_train_new, y_train)
score_2 = model_1.score(x_test_new, y_test)
print('Logistic Regression - Training Accuracy:', score_1)
print('Logistic Regression - Testing Accuracy:', score_2)

# (2) Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model_2 = DecisionTreeClassifier()
model_2.fit(x_train_new, y_train)

# Model evaluation for Decision Tree
s_1 = model_2.score(x_train_new, y_train)
s_2 = model_2.score(x_test_new, y_test)
print('Decision Tree - Training Accuracy:', s_1)
print('Decision Tree - Testing Accuracy:', s_2)

# (3) Support Vector Classifier
from sklearn.svm import SVC
model_3 = SVC()
model_3.fit(x_train_new, y_train)

# Model evaluation for SVC
s_a = model_3.score(x_train_new, y_train)
s_b = model_3.score(x_test_new, y_test)
print('SVC - Training Accuracy:', s_a)
print('SVC - Testing Accuracy:', s_b)

# Classification Reports for all 3 models
from sklearn.metrics import classification_report

p1 = model_1.predict(x_test_new)
print('1. Classification Report for Logistic Regression:', '\n', classification_report(y_test, p1))

pred1 = model_2.predict(x_test_new)
print('2. Classification Report for Decision Tree:', '\n', classification_report(y_test, pred1))

pred_a = model_3.predict(x_test_new)
print('3. Classification Report for SVC:', '\n', classification_report(y_test, pred_a))

# Save the best model
import pickle
pickle.dump(model_3, open('Model_for_SpamDetection.pkl', 'wb'))
print('Model saved as "Model_for_SpamDetection.pkl"')
