import os
from flask import Flask, request, jsonify, send_from_directory
from cslib import model, config

app = Flask(__name__)

def _test_flag(body):
    return bool(body.get("test", False))

@app.route("/train", methods=["POST"])
def train():
    body = request.get_json(silent=True) or {}
    try:
        out = model.train(data_dir=body.get("data_dir", config.DATA_DIR), test=_test_flag(body))
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True) or {}
    missing = [k for k in ("year", "month", "day") if k not in body]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400
    try:
        out = model.predict(body.get("country", "all"), body["year"], body["month"], body["day"],
                            data_dir=body.get("data_dir", config.DATA_DIR), test=_test_flag(body))
        return jsonify(out)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/logs/<path:filename>", methods=["GET"])
def logs(filename):
    test = request.args.get("test", "false").lower() == "true"
    d = config.log_dir(test)
    if not os.path.exists(os.path.join(d, filename)):
        return jsonify({"error": "file not found"}), 404
    return send_from_directory(d, filename)

@app.route("/logs", methods=["GET"])
def list_logs():
    test = request.args.get("test", "false").lower() == "true"
    return jsonify(sorted(os.listdir(config.log_dir(test))))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
