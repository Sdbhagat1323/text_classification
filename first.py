
# Importations
import pandas as pd
import numpy as np
from collections import Counter 
from sklearn.feature_extraction import DictVectorizer
import pickle
from sklearn.metrics import roc_auc_score, f1_score, log_loss, make_scorer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from joblib import dump, load
# importting all dataset 
train =  pd.read_csv("train.csv", engine = 'c')
train_label = pd.read_csv("trainLabels.csv", engine='c')
train_label.drop(train_label.index[9999:49999],0,inplace=True)
# now merging two files in one data set
train_with_labels = pd.merge(train, train_label, on='id')

# Now importing testing dataset
test =pd.read_csv("test.csv", engine="c")

# seems like test file don't have column name
column_list = []
for column in train.columns:
    column_list.append(column)

# Add column names to test file
test.columns = column_list


#### Data Wrangling and Feature encoding ###

# initializing empty list


vec = DictVectorizer()
names_categorical = []


# Encoding of dataset

train_with_labels.replace('YES', 1, inplace=True)
train_with_labels.replace('NO', 0, inplace = True)
train_with_labels.replace('nan', np.NaN, inplace = True)

# Encoding for test Dataset
test.replace('YES', 1, inplace=True)
test.replace('NO', 0, inplace = True)
test.replace('nan', np.NaN, inplace = True)

# checking all the columns types in data set
train_with_labels.columns.map(type)
#train_with_labels.columns.map(type)


'''
for name in train.columns:
    print(name.type)
    '''
# convert all columns name to string
train_with_labels.columns =train_with_labels.columns.astype(str)
test.columns = test.columns.astype(str)


# creating two numerical and categorial datasets



'''
#########################for some reason this is not workingg ##################
for name in train_with_labels.columns:
    #print(name)
    if name.startswith('x'):
        #count the datatype in each column
        #type_column = max(Counter(map(lambda x: str(type(x)), train_with_labels[name])).items(), key= lambada x:x[1])
        type_column, _ = max(Counter(map(lambda x: str(type(x)), train_with_labels[name])).items(), key = lambda x:x[1])

        
        if type_column == str(str):
            train_with_labels[name] = map(str, train_with_labels[name])
            test[name] = map(str, test)
            
            categorical_feature.append(name)
            
            print(name, len(np.unique(train_with_labels[name])))
            
            
        else:
            X_numerical.append(train_with_labels[name].fillna(-999))
            X_test_num.append(test[name].fillna(-999))
    
'''


numeric_train_data = train.select_dtypes(include=[np.number])
categorical_train_data = train.select_dtypes(exclude=[np.number])

numeric_test_data = test.select_dtypes(include=[np.number])
categorical_test_data = test.select_dtypes(exclude=[np.number])






        
         

for name in numeric_train_data.columns:
    # fill missing values with  -999
    numeric_train_data[name].fillna(-999)
    
for name in numeric_test_data.columns:
    
    numeric_test_data[name].fillna(-999)
    

##### Creating the numpy array from dataset #########

numeric_train_data = np.column_stack(numeric_train_data)
numeric_test_data = np.column_stack(numeric_test_data)





X_sparse = vec.fit_transform(train_with_labels[names_categorical].T.to_dict().values())
X_test_sparse = vec.transform(test[names_categorical].T.to_dict().values())

numeric_train_data = np.nan_to_num(numeric_train_data)
numeric_test_data = np.nan_to_num(numeric_test_data)

################### pickling data #####################

dump( (X_sparse, X_test_sparse, numeric_train_data,numeric_test_data), 
    'X.dump', compress = 1, )




######### Base Classifier Level #############

log_loss_scorer = make_scorer(log_loss, needs_proba = True)




y_columns = [name for name in train_with_labels.columns if name.startswith('y')]


X_numerical_base, X_numerical_meta, X_sparse_base, X_sparse_meta, y_base, y_meta = train_test_split(numeric_train_data, X_sparse, train_with_labels[y_columns].values,
        test_size = 0.2) # Note these are random splits 20/80









