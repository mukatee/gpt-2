__author__ = 'teemu kanstren'

from flask import Flask, request, jsonify
import boto3
import json

app = Flask(__name__)

@app.route('/api/generate_text/', methods = ['POST'])
def generate_text():
    import train2
    content = request.json
    user = content['user']
    seed_text = content['seed_text']
    generated_text = train2.generate_sample(seed_text)
    aws_tweet(user, generate_text)
    return jsonify({"generated_text": generated_text})

def aws_tweet(user, generated_text):
    generated_text = generated_text.replace("\\n", "\n")
    lb = boto3.client("lambda", region_name = 'eu-central-1')
    arn = 'arn:aws:lambda:eu-central-1:399551198609:function:LyricTweeter'
    param_data = {"user": str(user), "lyric": str(generated_text)}
    lb.invoke(FunctionName = arn, InvocationType = 'Event', Payload = bytes(json.dumps(param_data), encoding = "utf8"))


if __name__ == '__main__':
    #train2.generate_sample("bootstrap")
    app.run(host= '0.0.0.0',debug=True, port=5566)


