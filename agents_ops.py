
from dotenv import load_dotenv
from agents import Agent, Runner, trace, function_tool,input_guardrail, GuardrailFunctionOutput
from openai.types.responses import ResponseTextDeltaEvent
from typing import Dict,Literal
import os
import asyncio
from openai import OpenAI
from anthropic.types import TextBlock
import anthropic
import asyncio
from typing import List
from datetime import datetime

from pineconee import PineconeDocStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv(override=True)

pinecone = PineconeDocStore()



@function_tool
def query_from_db(query:str,namespace:Literal['gpt','claud','perplexity']):
    """ query the text from database based on user question """

    
    result = pinecone.search_query(query,namespace)
    return result

@function_tool
def save_to_db(namespace:Literal['gpt','claud','perplexity'],subject:str,text:str):
    """save the file in data base, choose a subject name for the text and send requested text to this function"""

    
    pinecone.upsert_all(namespace=namespace,subject=subject,text_src=text)
    return f'done with this subject{subject}'


@function_tool
def perpelexity_web_search(query:str):

    """perpelexity websearch perform the query based on requested search"""
    client = OpenAI(
        api_key=os.getenv("PERPLEXITY_API_KEY"),
        base_url="https://api.perplexity.ai"
    )
    

    response = client.chat.completions.create(
    model="sonar-pro",
    messages=[{"role":"user","content":query}],
    web_search_options={
        "search_context_size": "high", # depth of snippets
    }
    )

    return response.choices[0].message.content


@function_tool
def claud_web_search(query:str):
    """ claud web search perform the query based on requested search when user asked you with query you do indepth websearch for query"""
    
    antro = os.getenv('anthropic_api_key')
    client_cld = anthropic.Anthropic(api_key=antro)
    
    response = client_cld.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": f"{query}"}
        ],
        tools=[
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 1
            }
        ]
    )
    
    text_only = "".join(
        block.text                    # grab the words
        for block in response.content # iterate over blocks
        if isinstance(block, TextBlock)  # keep only TextBlocks
    )
    return text_only


@function_tool
def gpt_web_search(query: str) -> str:
    """GPT web search – performs a live web search via GPT models that have the
    built-in `web_search` tool. Returns the assistant’s answer with citations."""

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(

        model="gpt-4o-search-preview",

        messages=[{"role": "user", "content": query}],

        # Optional: fine-tune search depth, geography, etc.
        web_search_options={
            "search_context_size": "high",   # low | medium (default) | high

        }
    )

    # The answer (with citations inline) is in the assistant message:
    return response.choices[0].message.content


# simple Python prefix check – no extra agent needed
@input_guardrail
async def guardrail_requires_re(ctx, agent, message):
    last = message[-1]                      # assume the newest msg is last
    content = last.get("content", "") if isinstance(last, dict) else str(last)

    starts_with_re = content.lower().startswith("p")
    if starts_with_re:
        # allow the call to proceed
        return GuardrailFunctionOutput(output_info={"auth": "ok"}, tripwire_triggered=False)
    else:
        # block the call
        return GuardrailFunctionOutput(
            output_info={"auth": "failed"},
            tripwire_triggered=True
        )


