
def create_prompt(role, topic, audience, difficulty, language, output_format):
    prompt = f"""
Role:
{role}

Task:
Teach the topic: {topic}

Audience:
{audience}

Difficulty:
{difficulty}

Language:
Explain everything using {language}.

Instructions:
1. Explain the topic in simple language.
2. Include important concepts.
3. Give two practical examples.
4. Explain where this topic is used in real life.
5. Provide a real-world coding example related to the topic.
6. Include three quiz questions.

Output Format:
{output_format}
"""
    return prompt.strip()


def display_prompt(prompt):
    print("\n" + "=" * 50)
    print("GENERATED AI PROMPT")
    print("=" * 50)
    print(prompt)
    print("=" * 50)


def main():
    print("AI LEARNING PROMPT GENERATOR")
    print("-" * 30)

    role = input("Enter AI role: ")
    topic = input("Enter topic: ")
    audience = input("Enter audience: ")
    difficulty = input("Enter difficulty (Beginner/Intermediate/Advanced): ")
    language = input("Choose language (English/Filipino): ")

    print("\nChoose output format:")
    print("1. JSON")
    print("2. Plain Text")

    format_choice = input("Enter your choice (1 or 2): ")

    if format_choice == "1":
        output_format = """
        Return the response in valid JSON format.
        Use these keys:
        - definition
        - important_concepts
        - practical_examples
        - real_world_applications
        - coding_example
        - quiz
        """
    elif format_choice == "2":
        output_format = """
        Return the response in plain text.
        Use clear headings and numbered lists.
        """
    else:
        print("Invalid output format. Using Plain Text.")
        output_format = "Return the response in plain text."

    prompt = create_prompt(
        role,
        topic,
        audience,
        difficulty,
        language,
        output_format
    )

    display_prompt(prompt)


if __name__ == "__main__":
    main()
