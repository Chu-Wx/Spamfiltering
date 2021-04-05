{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "--2021-02-10 13:13:26--  https://docs.google.com/uc?export=download&id=1OVRo37agn02mc6yp5p6-wtJ8Hyb-YMXR\n",
      "Resolving docs.google.com (docs.google.com)... 2607:f8b0:4006:818::200e, 172.217.12.174\n",
      "Connecting to docs.google.com (docs.google.com)|2607:f8b0:4006:818::200e|:443... connected.\n",
      "HTTP request sent, awaiting response... 302 Moved Temporarily\n",
      "Location: https://doc-14-04-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/ohv0p54696982mlplsge3sg6308u7sqn/1612980750000/08752484438609855375/*/1OVRo37agn02mc6yp5p6-wtJ8Hyb-YMXR?e=download [following]\n",
      "Warning: wildcards not supported in HTTP.\n",
      "--2021-02-10 13:13:27--  https://doc-14-04-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/ohv0p54696982mlplsge3sg6308u7sqn/1612980750000/08752484438609855375/*/1OVRo37agn02mc6yp5p6-wtJ8Hyb-YMXR?e=download\n",
      "Resolving doc-14-04-docs.googleusercontent.com (doc-14-04-docs.googleusercontent.com)... 2607:f8b0:4006:802::2001, 142.250.64.97\n",
      "Connecting to doc-14-04-docs.googleusercontent.com (doc-14-04-docs.googleusercontent.com)|2607:f8b0:4006:802::2001|:443... connected.\n",
      "HTTP request sent, awaiting response... 200 OK\n",
      "Length: 503663 (492K) [text/csv]\n",
      "Saving to: ‘spam.csv’\n",
      "\n",
      "spam.csv            100%[===================>] 491.86K  --.-KB/s    in 0.09s   \n",
      "\n",
      "2021-02-10 13:13:27 (5.49 MB/s) - ‘spam.csv’ saved [503663/503663]\n",
      "\n"
     ]
    }
   ],
   "source": [
    "!wget 'https://docs.google.com/uc?export=download&id=1OVRo37agn02mc6yp5p6-wtJ8Hyb-YMXR' -O spam.csv "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/user/.pyenv/versions/3.7.5/lib/python3.7/site-packages/pandas/compat/__init__.py:117: UserWarning: Could not import the lzma module. Your installed Python is incomplete. Attempting to use lzma compression will result in a RuntimeError.\n",
      "  warnings.warn(msg)\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "df = pd.read_csv(\"spam.csv\", usecols=[\"v1\", \"v2\"], encoding='latin-1')\n",
    "# 1 - spam, 0 - ham\n",
    "df.v1 = (df.v1 == \"spam\").astype(\"int\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "val_size = int(df.shape[0] * 0.15)\n",
    "test_size = int(df.shape[0] * 0.15)\n",
    "\n",
    "train_size= int(df.shape[0]*0.7)\n",
    "df=df.sample(frac=1)\n",
    "valcol=df['v2'].tolist()[int(0.7*df.shape[0]):int(0.85*df.shape[0])]\n",
    "testcol=df['v2'].tolist()[int(0.85*df.shape[0]):]\n",
    "traincol=df['v2'].tolist()[:int(0.7*df.shape[0])]\n",
    "Array={'val_size':valcol,'test_size':testcol,'train_size':traincol}\n",
    "\n",
    "train_texts, train_labels = traincol, df['v1'].tolist()[:int(0.7*df.shape[0])]\n",
    "val_texts, val_labels     = valcol, df['v1'].tolist()[int(0.7*df.shape[0]):int(0.85*df.shape[0])]\n",
    "test_texts, test_labels   = testcol, df['v1'].tolist()[int(0.85*df.shape[0]):]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def preprocess_data(data):\n",
    "    import spacy as sc\n",
    "    import string\n",
    "    nlp = sc.load(\"en_core_web_sm\")\n",
    "    stopwords=nlp.Defaults.stop_words\n",
    "    List=[]\n",
    "    for i in data:\n",
    "      token = [token.text.lower() for token in nlp(i)]\n",
    "      List_without_sw=[j for j in token if not j in stopwords]\n",
    "      List.append(List_without_sw)\n",
    "\n",
    "    preprocessed_data = List\n",
    "    return preprocessed_data\n",
    "\n",
    "train_data = preprocess_data(train_texts)\n",
    "val_data = preprocess_data(val_texts)\n",
    "test_data = preprocess_data(test_texts)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "class Vectorizer():\n",
    "    def __init__(self, max_features):\n",
    "        self.max_features = max_features\n",
    "        self.vocab_list = None\n",
    "        self.token_to_index = None\n",
    "\n",
    "    def fit(self, dataset):\n",
    "        from collections import Counter\n",
    "        Commonlist=(words for line in dataset for words in line)\n",
    "        tokenlist = Counter(Commonlist)\n",
    "        Mostcommonword=tokenlist.most_common(max_features)\n",
    "        self.vocab_list=[i[0] for i in Mostcommonword]\n",
    "        self.token_to_index={value : index for index, value in enumerate(self.vocab_list)}\n",
    "          \n",
    "\n",
    "    def transform(self, dataset):\n",
    "        data_matrix = np.zeros((len(dataset), len(self.vocab_list)))\n",
    "        for line_index, line in enumerate(dataset):\n",
    "          for word_index, word in enumerate(line):\n",
    "            if word in self.vocab_list:\n",
    "              word_pos=dataset[line_index][word_index]\n",
    "              data_matrix[line_index,self.token_to_index[word_pos]]=1\n",
    "        \n",
    "        return data_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "max_features = 400\n",
    "vectorizer = Vectorizer(max_features=max_features)\n",
    "vectorizer.fit(train_data)\n",
    "X_train = vectorizer.transform(train_data)\n",
    "X_val = vectorizer.transform(val_data)\n",
    "X_test = vectorizer.transform(test_data)\n",
    "\n",
    "y_train = np.array(train_labels)\n",
    "y_val = np.array(val_labels)\n",
    "y_test = np.array(test_labels)\n",
    "\n",
    "vocab = vectorizer.vocab_list"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "# Define Logistic Regression model\n",
    "model = LogisticRegression(random_state=0, solver='liblinear')\n",
    "\n",
    "# Fit the model to training data\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Make prediction using the trained model\n",
    "y_train_pred = model.predict(X_train)\n",
    "y_val_pred = model.predict(X_val)\n",
    "y_test_pred = model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "def accuracy_score(y_true, y_pred): \n",
    "    # Calculate accuracy of the model's prediction\n",
    "    tp,tn,fp,fn=0,0,0,0\n",
    "    for i, j in zip(y_pred,y_true):\n",
    "      if i==j and i !=0:\n",
    "        tp+=1\n",
    "      elif i==j and i==0:\n",
    "        tn+=1\n",
    "      elif i-j==-1:\n",
    "        fn+=1\n",
    "      elif i-j==1:\n",
    "        fp+=1\n",
    "    \n",
    "    accuracy = (tn+tp)/(tp+tn+fp+fn)\n",
    "    return accuracy\n",
    "\n",
    "def f1_score(y_true, y_pred): \n",
    "    # Calculate F1 score of the model's prediction\n",
    "    tp,tn,fp,fn=0,0,0,0\n",
    "    for i, j in zip(y_pred,y_true):\n",
    "      if i==j and i !=0:\n",
    "        tp+=1\n",
    "      elif i==j and i==0:\n",
    "        tn+=1\n",
    "      elif i-j==-1:\n",
    "        fn+=1\n",
    "      elif i-j==1:\n",
    "        fp+=1\n",
    "    recall=tp/(tp+fn)\n",
    "    precision=tp/(tp+fp)\n",
    "\n",
    "    f1 = 2*(precision*recall)/(precision+recall)\n",
    "    return f1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training accuracy: 0.986, F1 score: 0.945\n",
      "Validation accuracy: 0.972, F1 score: 0.886\n",
      "Test accuracy: 0.978, F1 score: 0.920\n"
     ]
    }
   ],
   "source": [
    "print(f\"Training accuracy: {accuracy_score(y_train, y_train_pred):.3f}, \"\n",
    "      f\"F1 score: {f1_score(y_train, y_train_pred):.3f}\")\n",
    "print(f\"Validation accuracy: {accuracy_score(y_val, y_val_pred):.3f}, \"\n",
    "      f\"F1 score: {f1_score(y_val, y_val_pred):.3f}\")\n",
    "print(f\"Test accuracy: {accuracy_score(y_test, y_test_pred):.3f}, \"\n",
    "      f\"F1 score: {f1_score(y_test, y_test_pred):.3f}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
