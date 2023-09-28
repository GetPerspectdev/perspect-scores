import os
from flask import Flask, jsonify, request
from web_service import ScoreProcessor
import threading

app = Flask(__name__)
processor = ScoreProcessor()
threading.Thread(target=processor.listen, args=()).start()

@app.route("/profesionalism-api/health", methods=["GET"])
def health_check():
    return jsonify({"msg": "Slack Profesionalism API"})

@app.route("/profesionalism-api/calculate", methods=["GET"])
def get_design_patterns():
    args = request.args
    score_id = args['score_id']
    slack_key = args['slack_key']
    print(args)
    if score_id is None or slack_key is None:
        return jsonify({"msg": "Missing arguments"}), 400
    threading.Thread(target=processor.enqueue_score, args=(slack_key, score_id)).start()
    return jsonify({"msg": "Calculating Profesionalism started"})

if __name__ == "__main__":
    debug = os.environ.get("DEBUG", False)
    port = os.environ.get("PORT", 5050)
    app.run(host="0.0.0.0", port=port, debug=debug)