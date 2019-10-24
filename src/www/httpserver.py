from flask import Flask, render_template, request

app = Flask(__name__)


# Serve main index
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/execute/<model>/<question>')
def execute(model, question):
    

if __name__ == '__main__':
    app.run()
