import spacy
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, render_template, request

nlp = spacy.load("en_core_web_sm")

data = []
with open("SMSSpamCollection", "r", encoding="utf-8") as f:
    for line in f:
        label, message = line.split("\t", 1)
        doc = nlp(message)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
        data.append((label, " ".join(tokens)))

x = [message for label, message in data]
y = [label for label, message in data]

vectorizer = CountVectorizer()
x_vectorizer = vectorizer.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_vectorizer, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(x_train, y_train)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        user_message = request.form.get("message", "")
        doc = nlp(user_message)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
        process_message = " ".join(tokens)
        vector = vectorizer.transform([process_message])
        prediction = model.predict(vector)[0]

        if prediction == "ham":
            result = "Normal Message"
        else:
            result = "Spam Message"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
