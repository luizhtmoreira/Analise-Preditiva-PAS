import toml
import sys
import traceback
from supabase import create_client

def test_connection():
    try:
        print("--- Diagnostic Script Start ---")
        
        # Load secrets
        try:
            secrets = toml.load(".streamlit/secrets.toml")
            url = secrets["supabase"]["url"]
            key = secrets["supabase"]["key"]
            print(f"Successfully loaded secrets from .streamlit/secrets.toml")
            print(f"URL: {url}")
            print(f"Key length: {len(key)}")
        except Exception as e:
            print(f"Error loading secrets: {e}")
            return

        # Initialize client
        print("\nAttempting to initialize Supabase client...")
        try:
            client = create_client(url, key)
            print("Supabase client initialized successfully.")
        except Exception as e:
            print(f"Error initializing client: {e}")
            traceback.print_exc()
            return

        # Simple data fetch test (e.g., check auth settings or a dummy select)
        print("\nAttempting a basic API call (auth.get_settings)...")
        try:
            # We don't need a table, just check if API responds
            # get_user() is safer if we want to check if the key is valid for auth
            # but since we aren't logged in, we expect a 401 or similar, which confirms CONNECTION
            res = client.auth.get_user()
            print("API call finished.")
        except Exception as e:
            print(f"\nAPI Call failed with error: {e}")
            # Check for specific network or timeout errors
            if "timeout" in str(e).lower():
                print(">>> Diagnosis: Connection Timeout. Possible network/firewall issue.")
            elif "invalid api key" in str(e).lower():
                print(">>> Diagnosis: Invalid API Key.")
            else:
                print(">>> Full traceback for API failure:")
                traceback.print_exc()

    except Exception as e:
        print(f"Unexpected top-level error: {e}")
        traceback.print_exc()
    finally:
        print("\n--- Diagnostic Script End ---")

if __name__ == "__main__":
    test_connection()
