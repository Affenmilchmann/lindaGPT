from flask import Flask
import os

app = Flask(__name__)
html_str = """
<!DOCTYPE html>
<html>
    <head>
        <title>AnekGPT</title>
    </head>
    <body>
        <p>This text must be visible</p>
    </body>
</html>
"""

@app.route("/")
def hello():
    return html_str

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True,host='0.0.0.0',port=port)
