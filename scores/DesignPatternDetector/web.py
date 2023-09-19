import os
from flask import Flask, jsonify, request
from web_app import get_github

app = Flask(__name__)

@app.route("/patterns-api/health", methods=["GET"])
def health_check():
    return jsonify({"msg": "GH Design Patterns API"})

@app.route("/patterns-api/calculate", methods=["GET"])
def get_design_patterns():
    args = request.args
    github_repo = args['github_repo']
    dev_key = args.get('gh_key', '')
    branch = args.get('branch', 'main')
    debug = args.get('debug', False)
    print(args)
    return get_github(github_repo, dev_key, branch, debug)

if __name__ == "__main__":
    debug = os.environ.get("DEBUG", False)
    port = os.environ.get("PORT", 5050)
    app.run(host="0.0.0.0", port=port, debug=debug)