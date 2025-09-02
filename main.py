# main_tweet_qa_agent.py
# This file defines and runs a Tweet-Style Question Answering agent, a Sentiment Analysis agent, an English Tutor agent, and a Banking Assistant agent.

from agents import Agent, Runner
from connection import config, model

# Define the TweetAnswerer agent with multiple tasks.
tweet_qa_agent: Agent = Agent(
    name="TweetAnswerer",
    instructions="""
    You are a concise and witty question-answering assistant with two tasks:
    1. Answer general knowledge questions in a tweet-like format (1-2 sentences, end with #QuickFact).
    2. If the input includes 'fact-check:', verify the statement's accuracy and respond concisely with the result, ending with #FactCheck.
    Keep responses short and engaging.
    """,
    model=model
)

# Define the SentimentAnalyzer agent.
sentiment_analyzer_agent: Agent = Agent(
    name="SentimentAnalyzer",
    instructions="""
    You analyze the sentiment of user questions.
    Respond with a single sentence stating whether the sentiment is positive, negative, or neutral, ending with #SentimentAnalysis.
    """,
    model=model
)

# Define the EnglishTutor agent.
english_tutor_agent: Agent = Agent(
    name="EnglishTutor",
    instructions="""
    You are an English tutor providing concise lessons or corrections.
    If the input is a question about English (e.g., grammar, vocabulary, or usage), provide a brief explanation or example.
    If the input is a sentence, correct any English errors and explain the correction briefly, ending with #EnglishLesson.
    Keep responses short and clear.
    """,
    model=model
)

# Define the BankingAssistant agent.
banking_assistant_agent: Agent = Agent(
    name="BankingAssistant",
    instructions="""
    You are a banking assistant for First Citizen's Bank, answering basic banking-related questions.
    Provide concise answers about accounts, loans, or services in 1-2 sentences.
    Include 'Member FDIC. Terms apply.' and 'For details: firstcitizens.com or 1-800-FCB-HELP' in each response, ending with #BankingBasics.
    """,
    model=model
)

# Start the interactive session.
print("--- Welcome to the Tweet-Style Answer, Sentiment Analysis, English Tutor & Banking Assistant Agents! ---")
print("Ask any general knowledge question, use 'fact-check:' for verification, ask about English grammar/vocabulary, or pose banking-related questions.")
print("You can also provide a sentence for English correction.")
print("The SentimentAnalyzer will assess your question's sentiment.")
print("Type 'quit' or 'exit' to end the session.\n")

while True:
    try:
        # Get user input.
        user_question = input("Your question or sentence: ")
        
        if user_question.lower() in ["quit", "exit"]:
            print("\nGoodbye! Stay curious! #KnowledgeIsPower")
            break

        if not user_question.strip():
            continue
        
        print("\nProcessing your input...")
        
        # Run TweetAnswerer agent for answering or fact-checking.
        result_qa = Runner.run_sync(tweet_qa_agent, user_question, run_config=config)
        qa_response = result_qa.final_output

        # Run SentimentAnalyzer agent for sentiment analysis.
        result_sentiment = Runner.run_sync(sentiment_analyzer_agent, user_question, run_config=config)
        sentiment_response = result_sentiment.final_output

        # Run EnglishTutor agent for English lessons or corrections.
        result_english = Runner.run_sync(english_tutor_agent, user_question, run_config=config)
        english_response = result_english.final_output

        # Run BankingAssistant agent for banking-related questions.
        result_banking = Runner.run_sync(banking_assistant_agent, user_question, run_config=config)
        banking_response = result_banking.final_output

        print(f"\nAnswer: {qa_response}")
        print(f"Sentiment: {sentiment_response}")
        print(f"English Lesson: {english_response}")
        print(f"Banking Info: {banking_response}\n")
    
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Please restart the application or try a different input.")
        break