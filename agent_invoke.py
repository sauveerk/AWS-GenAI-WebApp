import boto3
import json
import os
from botocore.exceptions import ClientError
from typing import Optional, Dict, Any
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def invoke_bedrock_agent(prompt: str) -> Optional[Dict[Any, Any]]:
    """
    Invoke a Bedrock agent with the given prompt.
    
    Args:
        prompt (str): The prompt to send to the agent
        
    Returns:
        Optional[Dict]: The agent's response or None if an error occurs
    """
    try:
        # Get AWS region from environment
        region = os.getenv('AWS_DEFAULT_REGION')
        if not region or region == 'your_region':
            print("Error: AWS_DEFAULT_REGION not set in .env file")
            return None
            
        bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=region)
    except Exception as e:
        print(f"Error creating Bedrock client: {str(e)}")
        return None
    
    if not prompt or not isinstance(prompt, str):
        print("Error: Prompt must be a non-empty string")
        return None
        
    try:
        # Get agent configuration from environment
        agent_id = os.getenv('BEDROCK_AGENT_ID')
        agent_alias_id = os.getenv('BEDROCK_AGENT_ALIAS_ID')
        
        if not agent_id or agent_id == 'your_agent_id':
            print("Error: BEDROCK_AGENT_ID not set in .env file")
            return None
            
        if not agent_alias_id or agent_alias_id == 'your_agent_alias_id':
            print("Error: BEDROCK_AGENT_ALIAS_ID not set in .env file")
            return None
        
        session_id = str(uuid.uuid4())
        
        response = bedrock_agent_runtime.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            inputText=prompt
        )
        
        # Handle the event stream response
        full_response = ""
        
        for event in response['completion']:
            if 'chunk' in event:
                chunk = event['chunk']
                if 'bytes' in chunk:
                    chunk_data = chunk['bytes'].decode('utf-8')
                    full_response += chunk_data
        
        # Parse the complete response if needed
        if full_response:
            try:
                return json.loads(full_response)
            except json.JSONDecodeError:
                # If response is not JSON, return as plain text
                return {"response": full_response}
        else:
            print("Error: Empty response from Bedrock Agent")
            return None
            
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'throttlingException':
            return {"error": "Request rate too high. Please wait a moment and try again."}
        elif error_code == 'AccessDeniedException':
            return {"error": "Access denied. Please check your AWS permissions."}
        else:
            print(f"AWS API Error: {str(e)}")
            return {"error": f"AWS Error: {e.response['Error']['Message']}"}
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}"}

def main():
    # Prompt user for input
    prompt = input("Enter your prompt: ")
    # Call Bedrock agent with the prompt and get response
    response = invoke_bedrock_agent(prompt)

    # If we got a valid response back
    if response:
        print("Agent Response:")
        # For dictionary responses, pretty print as JSON
        if isinstance(response, dict):
            print(json.dumps(response, indent=2))
        # For non-dictionary responses, print directly
        else:
            print(response)
    # Handle case where we got no response
    else:
        print("Failed to get response from agent")

if __name__ == "__main__":
    main()
