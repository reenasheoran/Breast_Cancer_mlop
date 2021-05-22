from flask import Flask, render_template, request, jsonify
import os
import yaml
import joblib
import numpy as np

params_path="params.yaml"
webapp_root="webapp"

static_dir=os.path.join(webapp_root,"static")
template_dir=os.path.join(webapp_root,"templates")

app = Flask(__name__,static_folder=static_dir,template_folder=template_dir)

def read_params(config_path):
    with open(config_path) as yaml_file:
        config=yaml.safe_load(yaml_file)
    return config

def predict(data):
    config=read_params(params_path)
    model_dir_path=config["webapp_model_dir"]
    model = joblib.load(model_dir_path)
    pred = model.predict(data)
    print(pred)
    return pred[0]

def api_response(request):
    try:
        data = np.array([list(request.json.values())])
        response = predict(data)
        if response == 1:
            response='The cancer is Malignant'
        else:
             response='The cancer is Bengin'
        response = {"response": response}
        return response
    except Exception as e:
        print(e)
        error = {"error":"something went wrong!!! TRY AGAIN"}
        return error

@app.route('/',methods=["GET","POST"])
def execute():
    if request.method=="POST":
        try:
            if request.form:
                data=dict(request.form).values()
                data = [list(map(float,data))]
                response = predict(data)
                if response == 1:
                    response='The cancer is Malignant'
                else:
                    response='The cancer is Bengin'
                return render_template("index.html",response=response)

            elif request.json:
                response= api_response(request)
                return jsonify(response)

        except Exception as e:
            print(e)
            error= {"error":"Something went wrong!!! TRY AGAIN"}
            return render_template("404.html",error=error)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0",port = 5000,debug=True)