from flask import Flask, render_template, request
from agent_invoke import invoke_bedrock_agent
import json

app = Flask(__name__)

def extract_response_text(response):
    try:
        # Handle None response
        if response is None:
            return "No response received from Bedrock Agent"
            
        if isinstance(response, str):
            response_dict = json.loads(response)
        else:
            response_dict = response
            
        # Ensure response_dict is not None before checking keys
        if response_dict is None:
            return "No response received from Bedrock Agent"
            
        # Check for error field first
        if 'error' in response_dict:
            return response_dict['error']
        # Check for 'response' field in the dictionary
        elif 'response' in response_dict:
            return response_dict['response']
        # Check for 'completion' field as fallback
        elif 'completion' in response_dict:
            return response_dict['completion']
        # Return the full response if no known fields are found
        return str(response_dict)
    except Exception as e:
        return f"Error processing response: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def home():
    response = None
    if request.method == 'POST':
        user_prompt = request.form.get('prompt')
        if user_prompt:
            raw_response = invoke_bedrock_agent(user_prompt)
            response = extract_response_text(raw_response)
    return render_template('index.html', response=response)

if __name__ == '__main__':
    app.run(debug=True)
