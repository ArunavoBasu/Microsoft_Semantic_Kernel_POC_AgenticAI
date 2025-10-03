import asyncio
import streamlit as st
from os import environ

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.filters import FunctionInvocationContext

#Read environment variables
endpoint = environ["azure_end_point_url"]
api_key = environ["azure_openai_api_key"]
api_version = environ["azure_openai_api_version"]
deployment = environ["azure_model_name"]

#Initialize Azure chat completion connector
azure_chat = AzureChatCompletion(
    api_key=api_key,
    api_version=api_version,
    deployment_name=deployment,
    endpoint=endpoint
)


#Define the auto function invocation filter that will be used by the kernel
async def function_invocation_filter(context: FunctionInvocationContext, next):
    """A function that will be called for each function call in the response"""
    if 'messages' not in context.arguments:
        await next(context)
        return
    else:
        print(f"\n Agent [{context.function.name}] called with messages: {context.arguments['messages']}\n")
        await next(context)
        print(f"Response from agent [{context.function.name}]: {context.result.value}\n")
  

#Initialize Kernel
kernel = Kernel()

# The filter is used for demonstration purpose to show the function invocation
kernel.add_filter("function_invocation", function_invocation_filter)


#Defining Agents---
## Sentiment Analyzer Agent
sentiment_agent = ChatCompletionAgent(
    service=azure_chat,
    name="SentimentAgent",
    instructions='You are a sentiment analyzer expert. Analyze the user input and return the sentiment of the user input - Positive, Negetive or Neutral with a short explation'
)

## Keyword Extractor Agent
keyword_agent = ChatCompletionAgent(
    service=azure_chat,
    name="KeywordAgent",
    instructions='You are a keyword extractor agent. Extract top 5 relevant and most important keywords from the given input.'
)

## Name Entity Recognition Agent
ner_agent = ChatCompletionAgent(
    service=azure_chat,
    name="NERAgent",
    instructions='You are a Name Entity Recognition Agent(NER). Identify and list down all named entities in the input text, categorized by type(Person, Place, Organization, Location etc.)'
)

## Summarizer Agent
summarizer_agent = ChatCompletionAgent(
    service=azure_chat,
    name="SummarizerAgent",
    instructions='You are a Summarizer Agent. Identify and list down all named entities in the input text, categorized by type(Person, Place, Organization, Location etc.)'
)

# Orchestrator Agent - taking all agents as plugin
orchestrator_agent = ChatCompletionAgent(
    service=azure_chat,
    name= 'OrchestratorAgent',
    kernel=kernel,
    instructions= (
        "Your role is to evaluate the user's request and forward it to the appropriate agent based on the nature of"
        "the query. Forward requests about sentiment analysis to the SentimentAgent."
        "Forward requests concerning keyword extraction to the KeywordAgent."
        "Forward requests about name entity recognition such as person name, organization, location etc. to the NERAgent."
        "Forward requests about summarizing the input text to the SummarizerAgent."
        "Your goal is accurate identification of the appropriati specialist to ensure the "
        "user receives targeted assistance."
        "And generate a 'Analysis Report' that you extracted by all the agents"
    ),
    plugins=[sentiment_agent, keyword_agent, ner_agent, summarizer_agent]
)

# Creating Thread
thread: ChatHistoryAgentThread = None

# Main function
async def main() -> None:
    print(f"Welcome to the TEXT_AGENT_CHATBOT\n Type 'exit' to exit from the chatbot\n")
    
    while True:
        user_input = input("User:> ")
        if user_input.lower().strip() == 'exit':
            print("\n Exiting from the chat interface.....")
            return False
        else:
            response = await orchestrator_agent.get_response(
                messages=user_input,
                thread=thread
            )
        
        if response:
            print(f"\n Agent:> {response}")


if __name__ == "__main__":
    asyncio.run(main())