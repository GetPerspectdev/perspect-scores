import os
from flask import Flask, jsonify, request
from web_app import ds
from web_service import ScoreProcessor
import threading

app = Flask(__name__)
processor = ScoreProcessor()
threading.Thread(target=processor.listen, args=()).start()

@app.route("/patterns-api/health", methods=["GET"])
def health_check():
    return jsonify({"msg": "GH Design Patterns API"})

@app.route("/patterns-api/calculate", methods=["GET"])
def get_design_patterns():
    args = request.args
    score_id = args['score_id']
    github_repo = args['github_repo']
    branch = args.get('branch', 'main')
    debug = args.get('debug', False)
    print(args)
    if score_id is None or github_repo is None:
        return jsonify({"msg": "Missing arguments"}), 400
    threading.Thread(target=processor.enqueue_score, args=(github_repo, score_id, branch, debug, ds)).start()
    return jsonify({"msg": "Calculating Design Patterns started"})

if __name__ == "__main__":
    debug = os.environ.get("DEBUG", False)
    port = os.environ.get("PORT", 5050)
    app.run(host="0.0.0.0", port=port, debug=debug)