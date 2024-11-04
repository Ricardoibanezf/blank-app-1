import streamlit as st
import langchain
from openai import OpenAI
from langchain.llms import OpenAI
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

# Streamlit title
st.title("🛫 Airline Experience")

# User input
prompt = st.text_input("Share with us your experience of the latest trip.")

# Secret key for OpenAI API
my_secret_key = st.secrets["MyOpenAIKey"]
os.environ["openai_api_key"] = my_secret_key

# LLM Initialization
llm = ChatOpenAI(openai_api_key=my_secret_key, model="gpt-4o-mini")

# Prompt for classification
prompt_syst1 = """
You are a travel agent specializing in customer experience. Based on the provided text, determine:
1) "airline_negative" if the user had a negative experience due to the airline's fault.
2) "non_fault_airline_negative" if the user had a negative experience that was not the airline's fault.
3) "positive" if the user had a positive experience.

Provide only the result, with no additional explanation.

Text:
{experience_user}
"""

flight_chain = (
    PromptTemplate.from_template(prompt_syst1)
    | llm
    | StrOutputParser()
)

# Chains for each response type
positive_chain = PromptTemplate.from_template(
    """You are a travel agent specializing in customer experience. Thank the user for their feedback professionally. Respond in the first person and keep a conversational tone.

Text:
{text}
"""
) | llm

airline_negative_chain = PromptTemplate.from_template(
    """You are a travel agent specializing in customer experience. Show sympathy and inform the user that customer service will reach out soon.

Text:
{text}
"""
) | llm

non_airline_negative_chain = PromptTemplate.from_template(
    """You are a travel agent specializing in customer experience. Show sympathy but explain that the airline is not liable for this situation. Avoid mentioning compensation.

Text:
{text}
"""
) | llm

# Main chain template if no conditions match
main_chain = PromptTemplate.from_template(
    """You are a travel agent specializing in customer experience. Prompt the user: "I am here to help you, please provide your feedback."

Text:
{text}
"""
) | llm

# Branch to route based on experience type
branch = RunnableBranch(
    (lambda x: "airline_negative" in x["exp_type"].lower(), airline_negative_chain),
    (lambda x: "non_fault_airline_negative" in x["exp_type"].lower(), non_airline_negative_chain),
    (lambda x: "positive" in x["exp_type"].lower(), positive_chain),
    main_chain
)

# Combine chains
full_chain = {"exp_type": flight_chain, "text": lambda x: x["experience_user"]} | branch

# Debugging: Log the classification output
classification = flight_chain.invoke({"experience_user": prompt})
st.write("Classification result:", classification)  # Display the classification result for debugging

# Get the response based on classification
response = branch.invoke({"exp_type": classification, "experience_user": prompt})

# Display response
st.write(response.content)
