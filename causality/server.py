from flask import Flask, request, render_template, jsonify

from causality.bayesian_network import create_model_and_inference
from causality.graph import draw_network

import numpy as np


app = Flask(__name__)

model, inference = create_model_and_inference()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/nodes')
def nodes():
    return jsonify(model.nodes())


@app.route('/inference', methods=['POST'])
def infer():
    if request.method == 'POST':
        evidence = {i: 1 for i in request.get_json()}
        queried = np.setdiff1d(model.nodes(), request.get_json()).tolist()

        print('Starting inference')
        cpds = inference.query(variables=queried,
                               evidence=evidence)
        print('Finished inference')
        draw_network(model.nodes(), model.edges(), cpds, request.get_json())
        return 'OK'

    return 'NOK'


if __name__ == '__main__':
    app.run(port=8080, debug=False)
