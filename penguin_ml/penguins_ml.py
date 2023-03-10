# import necessary modules
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# read data 
penguin_df = pd.read_csv('penguins.csv')
penguin_df.dropna(inplace=True)
output = penguin_df['species']
features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm',
'flipper_length_mm', 'body_mass_g', 'sex']]

# one-hot encode categorical features
features= pd.get_dummies(features)
output, uniques = pd.factorize(output)

# split data into training and test set
x_train, x_test, y_train, y_test = train_test_split(features, 
output, test_size=0.8)

# train random forest classifier model
rfc = RandomForestClassifier(random_state=15)
rfc.fit(x_train, y_train)
y_pred= rfc.predict(x_test)
score= accuracy_score(y_pred, y_test)
print('Our accuracy score for this model is {}'.format(score))

#save model and labels to pickle files
rf_pickle= open('random_forest_penguin.pickle', 'wb')
pickle.dump(rfc, rf_pickle)
rf_pickle.close()

output_pickle= open('output_penguin.pickle', 'wb')
pickle.dump(uniques, output_pickle)
output_pickle.close()