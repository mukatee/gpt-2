__author__ = 'teemu kanstren'

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/generate_text/', methods = ['POST'])
def generate_text():
    import train2
    content = request.json
    seed_text = content['seed_text']
    generated_text = train2.generate_sample(seed_text)
    return jsonify({"generated_text": generated_text})

if __name__ == '__main__':
    #train2.generate_sample("bootstrap")
    app.run(host= '0.0.0.0',debug=True, port=5566)


