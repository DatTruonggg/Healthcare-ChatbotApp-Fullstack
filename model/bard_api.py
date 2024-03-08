from bardapi import BardCookies
from flask import Flask, request, jsonify

app = Flask(__name__)

cookie_dict = {
    "__Secure-1PSID": "eAi6MiqH2pg03y7RT2gvHCNzoRsLancfOIvOMLmY-s1jq3Of7mEVoNc-vUKNo_9dG1ggcw.",
    "__Secure-1PSIDTS": "sidts-CjIBPVxjSpKEcOXAgevOOJt3yHwQrn_-oXDywB2zB42a1l3Rmo92S46MQNbKxZ0iaLizChAA"
}

bard = BardCookies(cookie_dict=cookie_dict)


@app.route('/get_response', methods=['POST'])
def get_answer():
    data = request.get_json()
    answer = bard.get_answer(data['prompt'])
    length = len(answer)
    if length > 100:
        answer = answer[:100]
    return jsonify({'result': answer['content']})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
