from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/status", methods=["GET"])
def get_status():
    """Returns the operational status of the analytics service."""
    print("Request received for analytics status check.")
    response = {
        "status": "Python analytics service is running successfully"
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
