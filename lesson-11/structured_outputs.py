
import os
from typing import Literal

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field, ValidationError


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-4-hydrogen"


class FeedbackAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]

    reason: str = Field(
        description="Explain why the feedback has this sentiment."
    )

    priority: Literal["low", "medium", "high"]

    summary: str = Field(
        description="A short summary of the customer feedback."
    )


def create_client():
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY is missing.")

    return genai.Client(api_key=API_KEY)


def analyze_feedback(client, feedback):
    response = client.models.generate_content(
        model=MODEL,
        contents=f"Analyze this customer feedback: {feedback}",
        config={
            "response_mime_type": "application/json",
            "response_schema": FeedbackAnalysis,
            "max_output_tokens": 250
        }
    )

    result = FeedbackAnalysis.model_validate_json(response.text)

    return result


def display_result(result):
    print("\nCUSTOMER FEEDBACK ANALYSIS")
    print("==========================")

    print(result.model_dump_json(indent=2))

    print("\nANALYSIS SUMMARY")
    print("----------------")
    print("Sentiment:", result.sentiment)
    print("Reason:", result.reason)
    print("Priority:", result.priority)
    print("Summary:", result.summary)


def main():
    print("AI CUSTOMER FEEDBACK ANALYZER")
    print("=============================")

    try:
        client = create_client()

        feedback = input("\nEnter customer feedback: ").strip()

        if not feedback:
            print("Please enter customer feedback.")
            return

        print("\nAnalyzing feedback...")

        result = analyze_feedback(client, feedback)

        display_result(result)

    except ValueError as error:
        print("Configuration Error:", error)

    except ValidationError as error:
        print("Invalid AI response:", error)

    except Exception as error:
        print("API Error:", error)


if __name__ == "__main__":
    main()
