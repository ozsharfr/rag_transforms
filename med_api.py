import requests

# === Configuration ===
API_BASE_URL = "https://disease.sh/v3/covid-19/all"

# Sample LLM response (for testing)
llm_response = "Common symptoms of COVID-19 include fever, cough, and shortness of breath."

# === Step 1: Extract Entities (for demo, we fix the condition to COVID-19) ===
def extract_entities(user_query):
    """
    Extracts the condition and query type from the user query.
    """
    query_types = ["treatment", "symptoms", "diagnosis", "guidelines"]
    query_type = "symptoms"  # Default to 'symptoms' for this example

    for q_type in query_types:
        if q_type in user_query.lower():
            query_type = q_type
            break

    return "covid-19", query_type

# === Step 2: Construct API URL (fixed endpoint for this example) ===
def construct_api_url():
    """
    Constructs the API query URL.
    """
    return API_BASE_URL

# === Step 3: Query the API ===
def query_api(url):
    """
    Fetches data from the API.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.RequestException as e:
        print(f"API error: {e}")
        return {}

# === Step 4: Compare LLM Response with API Data ===
def compare_response(llm_response, api_data):
    """
    Compares the LLM response with the API data.
    """
    api_symptoms = "fever, cough, shortness of breath"
    if api_symptoms.lower() in llm_response.lower():
        return "LLM response is consistent with the latest data."
    
    return "LLM response is NOT consistent with the latest data."

# === Step 5: Main Function ===
def fact_check_response(user_query, llm_response):
    """
    Extracts entities, constructs query, fetches data, and compares responses.
    """
    condition, query_type = extract_entities(user_query)
    api_url = construct_api_url()
    api_data = query_api(api_url)
    
    # For demonstration, we use a fixed set of symptoms for COVID-19
    return compare_response(llm_response, api_data)

# === Example Usage ===
user_query = "What are the symptoms of COVID-19?"
result = fact_check_response(user_query, llm_response)
print(result)