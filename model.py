import pandas as pd
from matplotlib import style
style.use("ggplot")
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import re
import string
import nltk
nltk.download('stopwords')

from sklearn.svm import SVC
import pickle
data = pd.read_csv("labeled_data.csv")
print(data.head())

data["labels"] = data["class"].map({0: "Hate Speech",
                                    1: "Offensive Language",
                                    2: "No Hate and Offensive"})


data = data[["tweet", "labels"]]



stopword = stopwords.words('english')

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub(r"\@w+|\#",'',text)
    text = re.sub(r"[^\w\s]",'',text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    tweet_tokens = word_tokenize(text)
    filtered_tweets=[w for w in tweet_tokens if not w in stopword] #removing stopwords
    return " ".join(filtered_tweets)

nltk.download('punkt')
data.tweet=data['tweet'].apply(clean)

tweetData = data.drop_duplicates("tweet")



lemmatizer=WordNetLemmatizer()

nltk.download('wordnet')
tweetData.loc[:, 'tweet'] = tweetData['tweet'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))





tweetData['labels'].value_counts()
non_hate_tweets = tweetData[tweetData.labels=='No Hate and Offensive']
non_hate_tweets.head()
non_hate_tweets.value_counts()
text=''.join([word for word in non_hate_tweets['tweet']])
vect=TfidfVectorizer(ngram_range=(1,2)).fit(tweetData['tweet'])
feature_names=vect.get_feature_names_out()


vect=TfidfVectorizer(ngram_range=(1,3)).fit(tweetData['tweet'])
feature_names=vect.get_feature_names_out()


X = tweetData['tweet']
Y = tweetData['labels']
X = vect.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, Y_train)
pickle.dump(svm_model, open("model.pkl","wb"))