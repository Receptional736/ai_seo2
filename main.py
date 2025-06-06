from dotenv import load_dotenv
from agents import Agent, Runner, trace, function_tool,input_guardrail, GuardrailFunctionOutput
from openai.types.responses import ResponseTextDeltaEvent
from typing import Dict,Literal
import os
import asyncio
from openai import OpenAI
from anthropic.types import TextBlock
import anthropic
#pip install 'openai-agents[litellm]'
import gradio as gr
import asyncio
from typing import List
from datetime import datetime
from pinecone import Pinecone

from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv(override=True)
from agents_ops import query_from_db, save_to_db, perpelexity_web_search, claud_web_search,gpt_web_search,guardrail_requires_re






manager_preamble = """
You are the **Gatekeeper**.

start conversation with defining your capability

Goal  
• you have two seprate tase
1- brand visibility
    • Collect every detail the Worker will need to run a full brand-visibility audit.  
    • Only hand off when the information is complete and unambiguous for brand-visibility.
2- interacting with database
    • if use ask to query to db or put info to db use provided tools

set up for brand visibility task:
Ask the user—politely and succinctly—for:  
1. Brand name (e.g. “Receptional”)  
2. Canonical website URL (e.g. “https://www.receptional.com/”)  
3. One-sentence description of what the brand does and its main market (e.g. “UK-based PPC and SEO agency”).  
4. Preferred search engines or data sources, if the user cares (choose from: GPT, Claude, Perplexity).  
5. Optional: specific services or regions they want checked, competitor list, or any KPIs of interest.

Rules  
• Use bullet points and plain questions—no jargon.  
• If an item is missing, ask a follow-up that targets *only* that gap.  
• Confirm back the final set of answers in one neat summary:  
"""

worker_prompt = r"""
You are **Receptional’s senior SEO & brand-visibility analyst**.

╭───────────────────────────────────────────────╮
│ INPUT (from Gatekeeper)                       │
│   brand_name          │ e.g. “Receptional”    │
│   website_url         │ canonical URL         │
│   brand_summary       │ 1–2-sentence focus    │
│   preferred_sources   │ list[str]             │
│   extras              │ dict – may contain:   │
│                       │   • services          │
│                       │   • regions           │
│                       │   • competitors       │
│                       │   • example queries   │
╰───────────────────────────────────────────────╯

━━━━━━━━  1 · SMART DATA COLLECTION  ━━━━━━━━
For **each** preferred source you must issue *exactly* **3 tool calls with user choosen tools**
(unless a call fails → retry once, still ≤ 3).  
Use this pattern:

▸ **Call #1 – discovery**  
    • Ask broad prompts that cover brand identity and each service/region.  
    • Example prompt template:  
      ```
      {brand_name} quick audit:
      1) Who is {brand_name}?  
      2) Is {brand_name} reliable?  
      3) List the best {service} providers in {region}. Return JSON.
      ```

▸ **Call #2 – gap-fill**  
    • Read Call #1 answers; identify missing services, regions or competitor mentions.  
    • Craft *targeted* follow-ups (one compound prompt) that plug those gaps.  

▸ **Call #3 – confirm / citation sweep**  
    • Final prompt asking only for sources, URLs, or unclear items flagged “⚠️” so far.  


━━━━━━━  2 · SCORING & ANALYSIS  ━━━━━━━
• **Brand-recognition table** – sentiment (−5…+5), factual accuracy (✅/⚠️/❌), DA/DR of each cited domain.  
• **Visibility scoring** – 5 pts if brand in top-3, 3 pts pos 4-6, 1 pt elsewhere.  
• **Share-of-voice** – brand score ÷ total points for all brands in that query.  
• **Competitor delta** – brand score minus each competitor’s score.  

━━━━━━━  3 · OUTPUT (return Markdown *and* save)  ━━━━━━━
#### executive summary  
Key scores, biggest wins, critical gaps (≤ 120 words).

#### methodology  
| Source | Tokens used | Tool calls (should be 3) | Timestamp |

#### brand recognition  
| Source | Prompt | Answer (truncated) | Sentiment | Accuracy | Trust domain |

#### service visibility  
| Query | Source | Brands (ranked) | Brand pos. | Score | Share-of-voice |

#### competitor snapshot *(omit if none)*  
Bulleted strengths / weaknesses vs. each competitor.

#### recommendations  
Max 7, ranked high→low impact, each ≤ 20 words.

##### appendix A – raw answers  
Group by source → call # → full text.

##### appendix B – URLs scraped / cited  
Grouped by query.

━━━━━━━━  CONVENTIONS  ━━━━━━━━
• British English; sentence case headings.  
• Never skip the appendices; if empty, write “None found”.   
  and return the Markdown in chat.
"""



worker_agent = Agent(
        name="Professional Seo specialist for ai",
        instructions=worker_prompt,
        model="litellm/anthropic/claude-sonnet-4-20250514",
        tools=[gpt_web_search, perpelexity_web_search, claud_web_search]
)


manager_agent = Agent(
    name="Seo manager",
    model="gpt-4o",
    instructions=manager_preamble,
    handoffs=[worker_agent],
    tools=[save_to_db,query_from_db],
    input_guardrails=[guardrail_requires_re]
)




MAX_TURNS = 7                    # 5 full exchanges = 10 messages

async def process_input(message, history):
    history = history[-MAX_TURNS:] 
    history_formatted = []
    for user_ms, assistant_ms in history:
        history_formatted.append({"role": "user", "content": user_ms})
        history_formatted.append({"role": "assistant", "content": assistant_ms})

        
        
    messages =  history_formatted + [{"role": "user", "content": message}] 

    try:
        with trace('seo ai brand recognizer'):
            result = Runner.run_streamed(manager_agent, input=messages)
            response = ""
        
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    delta = event.data.delta
                    response += delta
                    yield response  # Stream response incrementally to the UI

    except Exception as e:
        # e.output_info is whatever you set in GuardrailFunctionOutput
        info = getattr(e, "output_info", {})
        friendly = info.get("auth", f"Auth failed{e}")
        yield friendly                    


if __name__ == '__main__':

    gr.ChatInterface(process_input, title="SEO Agent Chat").launch()