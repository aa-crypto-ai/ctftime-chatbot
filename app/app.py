from flask import Flask
from flask import render_template, request

from llm.llm import get_response

app = Flask(__name__)

@app.route('/chatbot', methods=["GET", "POST"])
def chatbot():

    if request.method == 'GET':

        prompt = ''

    if request.method == 'POST':

        prompt = request.form["prompt"]

    return render_template('chatbot.html', prompt=prompt)

# todo: figure out how grid.js can handle async requests for each model
# to be separated into llm's own routes, when it's of more complex project size
@app.route('/chatbot/inference', methods=["POST"])
def get_inference():
    # the available model names will be separated from this function
    models = [
        {'family_name': 'meta-llama', 'model_name': 'llama-3.2-1b-instruct:free'},
        {'family_name': 'meta-llama', 'model_name': 'llama-3.1-70b-instruct:free'},
        {'family_name': 'mistralai', 'model_name': 'mistral-small-24b-instruct-2501'},   # not free, skip during development
    ]

    prompt = request.form['prompt']
    if not prompt:
        return {'data': [], 'total_models': 0}

    data = []

    for model in models:
        family_name = model['family_name']
        model_name = model['model_name']
        text = get_response(prompt, family_name, model_name)
        data.append(
            {'family_name': family_name, 'model_name': model_name, 'output': text}
        )
    # todo: add evaluation results
    return {'data': data, 'total_models': len(data)}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
