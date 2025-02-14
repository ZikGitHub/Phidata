from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os

load_dotenv()

# web search agent  
web_search_agent = Agent(

    name="Web Search Agent",
    role="Search the web and collect information.",
    model= Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,

)

# Financial agent
finance_agent = Agent(
    name="Finance AI Agent",
    model= Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[
        YFinanceTools(stock_price=True, 
                      analyst_recommendations=True, 
                      stock_fundamentals=True,
                      company_news=True),
   
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)


multi_ai_agent= Agent(
    model= Groq(id="deepseek-r1-distill-llama-70b"),
    team=[web_search_agent, finance_agent],
    instructions=["Always include sources, Use Table to display the data"],
    show_tool_calls=True,
    markdown=True, 
)

multi_ai_agent.print_response("Summarize analyst recommendation and share the latest news for NVDA", stream=True)