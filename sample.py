import cohere

def check_cohere_api_key(api_key):
    try:
        co = cohere.Client(api_key)
        # Use a simple endpoint to test the key
        response = co.generate(
            model='command',
            prompt="Test",
            max_tokens=5
        )
        print("API key is valid and working.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    api_key = input("Enter your Cohere API key: ")
    check_cohere_api_key(api_key)