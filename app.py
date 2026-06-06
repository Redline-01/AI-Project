from flask import Flask, render_template, request
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


app = Flask(__name__)

data = pd.read_csv("students.csv")

pass_count = len(data[data["Pass"] == 1])
fail_count = len(data[data["Pass"] == 0])

plt.figure(figsize=(5,4))
plt.bar(
    ["Pass", "Fail"],
    [pass_count, fail_count]
)
plt.title("Students Results")
plt.ylabel("Number of Students")
plt.tight_layout()

plt.savefig("static/chart.png")
plt.close()

X = data[["Hours", "Absences"]]
y = data["Pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

history = []

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)


@app.route("/", methods=["GET", "POST"])
def home():

    result = ""
    confidence = ""

    if request.method == "POST":

        hours = int(request.form["hours"])
        absences = int(request.form["absences"])

        prediction = model.predict([[hours, absences]])

        probabilities = model.predict_proba(
            [[hours, absences]]
        )[0]


        if prediction[0] == 1:

            result = "✅ Studenti kalon"

            confidence = round(
                probabilities[1] * 100,
                2
            )

        else:

            result = "❌ Studenti nuk kalon"

            confidence = round(
                probabilities[0] * 100,
                2
            )

        history.append({
            "hours": hours,
            "absences": absences,
            "result": result,
            "confidence": confidence
        })

    table = data.to_html(
        classes="table table-striped table-bordered text-center",
        index=False,
        border=0
    )

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        accuracy=round(accuracy * 100, 2),
        table=table,
        pass_count=pass_count,
        fail_count=fail_count,
        history=history

    )

if __name__ == "__main__":
    app.run(debug=True)