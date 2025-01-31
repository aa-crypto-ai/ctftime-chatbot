from flask import Flask
from flask import render_template, request

from llm.llm import get_response

app = Flask(__name__)

@app.route('/chatbot', methods=["GET", "POST"])
def chatbot():

    # the available model names will be separated from this function
    models = [
        {'family_name': 'meta-llama', 'model_name': 'llama-3.2-1b-instruct:free'},
        {'family_name': 'meta-llama', 'model_name': 'llama-3.1-70b-instruct:free'},
        {'family_name': 'mistralai', 'model_name': 'mistral-small-24b-instruct-2501'},   # not free, skip during development
    ]

    if request.method == 'GET':

        prompt = ''

        for model in models:
            model['output'] = 'N/A'

    if request.method == 'POST':

        prompt = request.form["prompt"]

        for model in models:
            text = get_response(prompt, model['family_name'], model['model_name'])
            if text is None:
                text = '[Internal Error] please try again'
            # todo: need to cater for how to display new line in the output in frontend
            model['output'] = text.replace('"', "'").replace('\n', '\\n')   # dirty way for now to avoid quotes / newline crashing the javascript in grid.js

    return render_template('chatbot.html', models=models, prompt=prompt)

# this function should be used, to do inference per model instead of as a whole
# todo: figure out how grid.js can handle async requests for each model
# to be separated into llm's own routes, when it's of more complex project size
@app.route('/chatbot/inference', methods=["POST"])
def get_inference():
    prompt = 'what is CTF?'
    family_name = 'meta-llama'
    model_name = 'llama-3.2-1b-instruct:free'
    text = get_response(prompt, family_name, model_name)
    # todo: add evaluation results
    return text

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
