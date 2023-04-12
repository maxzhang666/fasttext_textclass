# Import necessary libraries
from flask import Flask, request, app, jsonify

# Create Flask app instance
ap = Flask(__name__)


# Define API endpoint
@ap.route('/api')
def my_api():
    # Get data from request
    data = request.get_json()

    # Process data
    result = process_data(data)

    # Return result as JSON
    return jsonify(result)


@ap.route('/')
def hello_world():
    return 'Hello World!'


# Define function to process data
def process_data(data):
    # Implement data processing logic here
    return data


if __name__ == '__main__':
    # Run app
    ap.run()
