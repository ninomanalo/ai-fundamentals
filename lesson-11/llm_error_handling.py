
import os
import time
import logging

from dotenv import load_dotenv
from google import genai
from google.genai import errors
from google.genai import types


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-4-hydrogen"

MAX_RETRIES = 3


logging.basicConfig(
    filename="ai_assistant.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_client():
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY is missing.")

    return genai.Client(
        api_key=API_KEY,
        http_options=types.HttpOptions(timeout=30000)
    )


def ask_ai(client, question, role, language):

    prompt = f"""
Role:
{role}

Language:
{language}

Instructions:
1. Answer clearly and simply.
2. Maximum 5 sentences.
3. Include one practical example if useful.

Question:
{question}
"""

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config={
                    "max_output_tokens": 250,
                    "temperature": 0.5
                }
            )

            if not response.text:
                raise ValueError("The AI returned an empty response.")

            logging.info("AI response generated successfully.")

            return response.text

        except errors.APIError as error:

            logging.error(
                "API error %s: %s",
                error.code,
                error.message
            )

            if error.code in [408, 429, 500, 502, 503, 504]:

                if attempt < MAX_RETRIES:
                    wait_time = 2 ** attempt

                    print(
                        f"Temporary API error. Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                else:
                    raise RuntimeError(
                        "Maximum retry attempts reached."
                    ) from error

            else:
                raise RuntimeError(
                    f"API request failed: {error.message}"
                ) from error

        except ValueError:
            raise

        except Exception as error:
            logging.exception("Unexpected error occurred.")
            raise RuntimeError(
                "An unexpected error occurred."
            ) from error


def main():

    print("RELIABLE AI LEARNING ASSISTANT")
    print("==============================")

    try:
        client = create_client()

        role = input("\nEnter AI role: ").strip()

        if not role:
            role = "You are an experienced Python instructor."

        print("\nChoose language:")
        print("1. English")
        print("2. Filipino")

        language_choice = input("Enter choice: ").strip()

        language = "Filipino" if language_choice == "2" else "English"

        print("\nAI Assistant is ready!")
        print("Type 'exit' to quit.")

        while True:

            question = input("\nYou: ").strip()

            if question.lower() == "exit":
                print("AI: Goodbye! Happy learning!")
                break

            if not question:
                print("Please enter a question.")
                continue

            try:
                print("\nAI is thinking...")

                answer = ask_ai(
                    client,
                    question,
                    role,
                    language
                )

                print("\nAI:", answer)

            except RuntimeError as error:
                print("\nError:", error)

            except ValueError as error:
                print("\nResponse Error:", error)

    except ValueError as error:
        print("\nConfiguration Error:", error)

    except Exception as error:
        print("\nApplication Error:", error)


if __name__ == "__main__":
    main()
