from flask import Flask, request, jsonify
from flask_cors import CORS
from fuzzy_engine import diagnose as fuzzy_diagnose
from ml_model    import ml_predict, get_model_info, get_tree_rules
from hybrid      import hybrid_diagnose

app = Flask(__name__)
CORS(app)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status":"ok","system":"CKD Hybrid Expert System","engines":["Fuzzy Logic","Decision Tree ML"]})

@app.route('/api/diagnose', methods=['POST'])
def run_hybrid_diagnosis():
    try:
        data = request.get_json()
        _validate(data)
        result = hybrid_diagnose(float(data['gfr']),float(data['creatinine']),float(data['protein']),float(data['bp']),float(data['sugar']),float(data['bun']))
        return jsonify({"success":True,"result":result})
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}),500

@app.route('/api/fuzzy', methods=['POST'])
def run_fuzzy_only():
    try:
        data = request.get_json()
        _validate(data)
        result = fuzzy_diagnose(float(data['gfr']),float(data['creatinine']),float(data['protein']),float(data['bp']),float(data['sugar']),float(data['bun']))
        return jsonify({"success":True,"result":result})
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}),500

@app.route('/api/ml', methods=['POST'])
def run_ml_only():
    try:
        data = request.get_json()
        _validate(data)
        result = ml_predict(float(data['gfr']),float(data['creatinine']),float(data['protein']),float(data['bp']),float(data['sugar']),float(data['bun']))
        return jsonify({"success":True,"result":result})
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}),500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    return jsonify({"ml_model":get_model_info(),"tree_rules":get_tree_rules(max_depth=4)})

@app.route('/api/test-cases', methods=['GET'])
def test_cases():
    cases = [
        {"name":"مريض طبيعي",     "gfr":105,"creatinine":0.9,"protein":12,  "bp":112,"sugar":85, "bun":12},
        {"name":"مرحلة 1",        "gfr":95, "creatinine":1.1,"protein":90,  "bp":120,"sugar":95, "bun":15},
        {"name":"مرحلة 2",        "gfr":72, "creatinine":1.5,"protein":180, "bp":130,"sugar":120,"bun":22},
        {"name":"مرحلة 3",        "gfr":48, "creatinine":2.2,"protein":350, "bp":145,"sugar":160,"bun":50},
        {"name":"مرحلة 4",        "gfr":22, "creatinine":4.0,"protein":1000,"bp":158,"sugar":195,"bun":85},
        {"name":"فشل كلوي",       "gfr":8,  "creatinine":9.0,"protein":3000,"bp":182,"sugar":240,"bun":145},
        {"name":"GFR طبيعي+حرج", "gfr":97, "creatinine":1.5,"protein":3550,"bp":163,"sugar":60, "bun":169},
    ]
    return jsonify({"cases":cases})

def _validate(data):
    for f in ['gfr','creatinine','protein','bp','sugar','bun']:
        if f not in data: raise ValueError(f"Missing: {f}")

if __name__ == '__main__':
    print("CKD Hybrid Expert System running on :5000")
   import os
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)