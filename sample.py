try:
    from linebot.v3.messaging import MessagingApiClient
    print("Successfully imported MessagingApiClient from linebot.v3.messaging")
    print(f"MessagingApiClient type: {type(MessagingApiClient)}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print("\n--- Checking linebot.v3.messaging module content ---")
import linebot.v3.messaging
print(dir(linebot.v3.messaging))