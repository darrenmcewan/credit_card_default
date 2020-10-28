import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
      
      
df = pd.read_csv('creditCardData.csv')
df.head(10)

#There is no missing data! 
print(df.isnull().sum())        
print(df.isnull().sum().sum())

for col in df:
  if not pd.api.types.is_numeric_dtype(df[col]):
    df = df.join(pd.get_dummies(df[col], prefix=col))

df.head()

y = df['default payment next month'] # Label
X = df.drop(columns=['default payment next month', 'ID']) # Features
X = X.select_dtypes(np.number)
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

# Predict the labels for test dataset
y_pred = clf.predict(X_test)

output_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred,})
output_df.head(50)


df = df.rename(columns={'default payment next month': 'default'})
df['default'] = df['default'].apply(lambda x: 'yes' if x==1 else 'no')

df.head()

# Import scikit-learn metrics module. See complete list of Classification metrics here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
from sklearn import metrics

from matplotlib import pyplot as plt
cm = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(cm)
cm_display.plot(values_format='d')
plt.show()


df.head()
# Several of these metrics have to work off of dummy codes rather than categorical values. Therefore:
y_test_dummies = pd.get_dummies(y_test)
y_pred_dummies = pd.get_dummies(y_pred)

print(cm[0,1])


# Accuracy  = (true positives + true negatives) / (total cases); ranges from 0 (worst) to 1 (best)
print(f"Accuracy:\t{metrics.accuracy_score(y_test, y_pred)}")

# Precision = (true positives / (true positives + false positives))
print(f"Precision:\t{metrics.precision_score(y_test, y_pred, labels=['0', '1'])}")

# Recall    = (true positives / (true positives + false negatives)) 
print(f"Recall:\t\t{metrics.recall_score(y_test, y_pred, labels=['no', 'yes'])}")

# F1= (2 * (precision * recall) / (precision + recall))
print(f"F1:\t\t{metrics.f1_score(y_test, y_pred, labels=['no', 'yes'])}")





# Average precision-recall score and curve
average_precision = metrics.average_precision_score(y_test_dummies, y_pred_dummies)
disp = metrics.plot_precision_recall_curve(clf, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))
plt.show()


# Classification Report
df_report = pd.DataFrame(metrics.classification_report(y_test, y_pred, output_dict=True))
df_report

