import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_ds_agent(tools):
    """
    Creates and returns the Data Science Research Agent executor.
    """
    # 1. Setup the Brain (LLM)
    # It automatically looks for the 'OPENAI_API_KEY' environment variable
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    
    # 2. Design the Persona (Prompt Engineering)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Data Science Research Assistant at TU Dortmund. "
                   "Your job is to help users analyze data, write clean Python code, "
                   "and generate structured Markdown research insights. "
                   "You have access to tools. Use them when necessary."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # 3. Construct the Agent Pipeline
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # 4. Return the Executor (the loop that handles running tools and reasoning)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor
