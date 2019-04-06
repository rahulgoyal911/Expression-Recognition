#!/usr/bin/env python
from flask import Flask, jsonify, request
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

languages = [{'name':'javascript'},{'name':'python'}]

@app.route('/', methods=['GET'])
def test():
    return jsonify({'message': 'It works'})


@app.route('/lang', methods=['POST'])
def test1():
	request.get_json(force=True)
	global languages
	a = request.json['name']
	language = {'name' : a}
	# print(language)
	res = "getData(a)"
	# languages.append(language)
	language = {'name':res}
	languages=language
	response = jsonify({'languages': languages})
	response.headers.add('Access-Control-Allow-Origin', '*')
	return response
    
    
@app.route('/aaa/<string:condition>', methods=['GET'])
def ow2(condition):
	aa = condition
	
	return jsonify({'message': "heeee"})
    
# function that will return string of expressions
def getData(charStr):
	print("function called")
	# store charstr as .png or .jpeg or .jpg

	# call ml function with that image as input

	# return the string

if __name__ == '__main__':
    app.run(debug=True)
