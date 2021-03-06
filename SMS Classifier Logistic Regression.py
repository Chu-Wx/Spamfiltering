!wget 'https://docs.google.com/uc?export=download&id=1OVRo37agn02mc6yp5p6-wtJ8Hyb-YMXR' -O spam.csv
import pandas as pd
import numpy as np

df = pd.read_csv("spam.csv", usecols=["v1", "v2"], encoding='latin-1')
# 1 - spam, 0 - ham
df.v1 = (df.v1 == "spam").astype("int")
val_size = int(df.shape[0] * 0.15)
test_size = int(df.shape[0] * 0.15)

train_size= int(df.shape[0]*0.7)
df=df.sample(frac=1)
valcol=df['v2'].tolist()[int(0.7*df.shape[0]):int(0.85*df.shape[0])]
testcol=df['v2'].tolist()[int(0.85*df.shape[0]):]
traincol=df['v2'].tolist()[:int(0.7*df.shape[0])]
Array={'val_size':valcol,'test_size':testcol,'train_size':traincol}

train_texts, train_labels = traincol, df['v1'].tolist()[:int(0.7*df.shape[0])]
val_texts, val_labels     = valcol, df['v1'].tolist()[int(0.7*df.shape[0]):int(0.85*df.shape[0])]
test_texts, test_labels   = testcol, df['v1'].tolist()[int(0.85*df.shape[0]):]

def preprocess_data(data):
    import spacy as sc
    import string
    nlp = sc.load("en_core_web_sm")
    stopwords=nlp.Defaults.stop_words
    List=[]
    for i in data:
      token = [token.text.lower() for token in nlp(i)]
      List_without_sw=[j for j in token if not j in stopwords]
      List.append(List_without_sw)

    preprocessed_data = List
    return preprocessed_data

train_data = preprocess_data(train_texts)
val_data = preprocess_data(val_texts)
test_data = preprocess_data(test_texts)

class Vectorizer():
    def __init__(self, max_features):
        self.max_features = max_features
        self.vocab_list = None
        self.token_to_index = None

    def fit(self, dataset):
        from collections import Counter
        Commonlist=(words for line in dataset for words in line)
        tokenlist = Counter(Commonlist)
        Mostcommonword=tokenlist.most_common(max_features)
        self.vocab_list=[i[0] for i in Mostcommonword]
        self.token_to_index={value : index for index, value in enumerate(self.vocab_list)}
          

    def transform(self, dataset):
        data_matrix = np.zeros((len(dataset), len(self.vocab_list)))
        for line_index, line in enumerate(dataset):
          for word_index, word in enumerate(line):
            if word in self.vocab_list:
              word_pos=dataset[line_index][word_index]
              data_matrix[line_index,self.token_to_index[word_pos]]=1
        
        return data_matrix

max_features = 400
vectorizer = Vectorizer(max_features=max_features)
vectorizer.fit(train_data)
X_train = vectorizer.transform(train_data)
X_val = vectorizer.transform(val_data)
X_test = vectorizer.transform(test_data)

y_train = np.array(train_labels)
y_val = np.array(val_labels)
y_test = np.array(test_labels)

vocab = vectorizer.vocab_list

from sklearn.linear_model import LogisticRegression

# Define Logistic Regression model
model = LogisticRegression(random_state=0, solver='liblinear')

# Fit the model to training data
model.fit(X_train, y_train)

# Make prediction using the trained model
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

def accuracy_score(y_true, y_pred): 
    # Calculate accuracy of the model's prediction
    tp,tn,fp,fn=0,0,0,0
    for i, j in zip(y_pred,y_true):
      if i==j and i !=0:
        tp+=1
      elif i==j and i==0:
        tn+=1
      elif i-j==-1:
        fn+=1
      elif i-j==1:
        fp+=1
    
    accuracy = (tn+tp)/(tp+tn+fp+fn)
    return accuracy

def f1_score(y_true, y_pred): 
    # Calculate F1 score of the model's prediction
    tp,tn,fp,fn=0,0,0,0
    for i, j in zip(y_pred,y_true):
      if i==j and i !=0:
        tp+=1
      elif i==j and i==0:
        tn+=1
      elif i-j==-1:
        fn+=1
      elif i-j==1:
        fp+=1
    recall=tp/(tp+fn)
    precision=tp/(tp+fp)

    f1 = 2*(precision*recall)/(precision+recall)
    return f1

print(f"Training accuracy: {accuracy_score(y_train, y_train_pred):.3f}, "
      f"F1 score: {f1_score(y_train, y_train_pred):.3f}")
print(f"Validation accuracy: {accuracy_score(y_val, y_val_pred):.3f}, "
      f"F1 score: {f1_score(y_val, y_val_pred):.3f}")
print(f"Test accuracy: {accuracy_score(y_test, y_test_pred):.3f}, "
      f"F1 score: {f1_score(y_test, y_test_pred):.3f}")
