
import os
from dotenv import load_dotenv
from google import genai


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-4-hydrogen"


def create_client():
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY is missing from your .env file.")

    return genai.Client(api_key=API_KEY)


def ask_ai(client, question, role, language):
    prompt = f"""
Role:
{role}

Task:
Answer the user's question.

Language:
Respond in {language}.

Instructions:
1. Keep your answer short and clear.
2. Use simple language.
3. Maximum 5 sentences.
4. Include one practical example if useful.
5. Avoid unnecessary explanations.

Question:
{question}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={
            "max_output_tokens": 250,
            "temperature": 0.5
        }
    )

    return response.text


def main():
    print("AI LEARNING ASSISTANT")
    print("=====================")

    try:
        client = create_client()

        role = input("\nEnter AI role: ").strip()

        if not role:
            role = "You are an experienced AI instructor."

        print("\nChoose your language:")
        print("1. English")
        print("2. Filipino")

        language_choice = input("Enter choice (1 or 2): ").strip()

        if language_choice == "2":
            language = "Filipino"
        else:
            language = "English"

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

            print("\nAI is thinking...")

            try:
                answer = ask_ai(
                    client,
                    question,
                    role,
                    language
                )

                print("\nAI:", answer)

            except Exception as error:
                print(f"\nAPI Error: {error}")

    except ValueError as error:
        print(f"Configuration Error: {error}")

    except Exception as error:
        print(f"Application Error: {error}")


if __name__ == "__main__":
    main()
