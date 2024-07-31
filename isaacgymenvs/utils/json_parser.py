import json
import sys

def extract_json_value(json_str, query):
    try:
        # Load the JSON data
        data = json.loads(json_str)
        
        # Split the query path by '.' and navigate through the JSON structure
        keys = query.split('.')
        for key in keys:
            if not key:  # Skip empty keys resulting from trailing or double dots
                continue
            if isinstance(data, list):
                try:
                    key = int(key)
                except ValueError:
                    print(f"Error: Expected an integer index for list access, got '{key}'")
                    return None
            try:
                data = data[key]
            except (KeyError, IndexError, TypeError):
                print(f"Error: Key '{key}' not found")
                return None
        
        # Return the extracted value
        return data
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON data - {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py '<json_str>' '<query>'")
        sys.exit(1)
    
    json_str = sys.argv[1]
    query = sys.argv[2]
    
    result = extract_json_value(json_str, query)
    if result is not None:
        if isinstance(result, str):
            print(result)  # Print string without quotes
        else:
            print(json.dumps(result, indent=4))  # Use json.dumps for other types to format nicely
