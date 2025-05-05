import os

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


def generate():
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    system_instruction = """You are an AI assistant that finds news related to Advertising and Marketing companies. 

    Here are some guidelines to follow:
    - Be helpful and informative.
    - Use company web sites, social media sites, and news sites to find news.
    - Priortize news and data from earnings calls and new product launches.
    - Make special note of where companies are investing in new technologies and services.
    - Do not provide any information that is not in English.
    - Do not provide any information that is not safe for work.
    - Do not provide any information that is not appropriate for all ages.
    - Be concise and to the point.
    - Focus on recent news, within the last few weeks.
    - If there is no news, say so.
    - If there is news, provide a summary of the news.
    - If there is news, provide a link to the source of the news.
    - If there is news, provide the date of the news.
    - If there is news, provide the title of the news.
    """

    model = "gemini-2.0-flash"
    # model = "gemini-2.5-pro-preview-03-25"


    # Initialize conversation history with the system instruction as the first user message
    conversation_history = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=system_instruction)],
        )
    ]
    tools = [types.Tool(google_search=types.GoogleSearch())]

    generate_content_config = types.GenerateContentConfig(
        temperature=0.5,
        response_mime_type="text/plain",
    )

    print("Chatbot: Hello! I'm your Amazon news assistant. How can I help you today?")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        # Add user input to conversation history
        conversation_history.append(
            types.Content(
                role="user", parts=[types.Part.from_text(text=user_input)]
            )
        )

        # Generate response based on the entire conversation history.
        try:
            response_stream = client.models.generate_content_stream(
                model=model,
                contents=conversation_history,
                config=generate_content_config,
            )

            full_response = ""
            print("Chatbot: ", end="")
            for chunk in response_stream:
                print(chunk.text, end="", flush=True)
                full_response += chunk.text
            print()

            # Add the chatbot's response to the conversation history
            conversation_history.append(
                types.Content(
                    role="model", parts=[types.Part.from_text(text=full_response)]
                )
            )
        except Exception as e:
            print(
                f"Chatbot: An error occurred: {e}"
            )  # Print the error message.
            print("Chatbot: Please try again.")  # Ask the user to try again.


if __name__ == "__main__":
    generate()
