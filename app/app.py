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
    # llama is very unstable in OpenRouter
    models = [
        # {'display_name': 'Meta Llama 3.2 1B', 'family_name': 'meta-llama', 'model_name': 'llama-3.2-1b-instruct:free'},
        # {'display_name': 'Meta Llama 3.2 3B', 'family_name': 'meta-llama', 'model_name': 'llama-3.2-3b-instruct:free'},
        {'display_name': 'Meta Llama 3.1 70B', 'family_name': 'meta-llama', 'model_name': 'llama-3.1-70b-instruct:free'},
        # below are all not free, skip during development
        {'display_name': 'Google Gemini 2.0', 'family_name': 'google', 'model_name': 'gemini-2.0-flash-001'},
        {'display_name': 'Claude 3.5 Sonnet', 'family_name': 'anthropic', 'model_name': 'claude-3.5-sonnet'},
        {'display_name': 'Mistral Small 24B', 'family_name': 'mistralai', 'model_name': 'mistral-small-24b-instruct-2501'},
        # openai not accessible in some regions
        # {'display_name': 'OpenAI ChatGPT 4o', 'family_name': 'openai', 'model_name': 'gpt-4o-2024-11-20'},
    ]

    prompt = request.form['prompt']
    if not prompt:
        return {'data': [], 'total_models': 0}

    data = []

    for model in models:
        family_name = model['family_name']
        model_name = model['model_name']
        display_name = model['display_name']
        result = get_response(prompt, family_name, model_name)
        if result is None:
            docs = None
            data.append(
                {
                    'family_name': family_name,
                    'model_name': model_name,
                    'display_name': display_name,
                    'docs': [],
                    'output': 'null',
                    'context_precision': None,
                }
            )
        else:
            docs = result['docs']
            data.append(
                {
                    'family_name': family_name,
                    'model_name': model_name,
                    'display_name': display_name,
                    'docs': result['docs'],
                    'output': result['response'],
                    'context_precision': result['context_precision'],
                }
            )
    # todo: add evaluation results
    return {'data': data, 'total_models': len(data), 'docs': docs}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
