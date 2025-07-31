from typing_extensions import TypedDict, Annotated

from langchain.chat_models import init_chat_model
from langchain.tools import Tool
from langchain_community.tools import tool
from langchain_core.messages import HumanMessage, AnyMessage

from langchain_community.utilities import SerpAPIWrapper
from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper
from langchain_experimental.tools.python.tool import PythonREPLTool 

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
alphavantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")


llm = init_chat_model("gpt-4.1", temperature=0.7)

serpapi_tool = Tool(
    name="serpapi",
    description="Optimizes e-commerce purchase flows by analyzing real-time search engine results for product visibility, competitor pricing, and market trends, improving conversion rates.",
    func=SerpAPIWrapper().run,
)

alpha_vantage_api = AlphaVantageAPIWrapper(alphavantage_api_key=alphavantage_api_key)

@tool
def currency_tool(from_currency: str, to_currency: str) -> str:
    """Get exchange rate between two currencies."""
    return alpha_vantage_api.run(from_currency=from_currency, to_currency=to_currency)

repl_tool = PythonREPLTool()


def mortgage_calculator(loan, rate, years):
    r = rate / 100 / 12
    n = years * 12
    payment = loan * r * (1 + r)**n / ((1 + r)**n - 1)
    return f"Monthly Payment: ${payment:.2f}"

mortgage_tool = Tool.from_function(
    func=mortgage_calculator,
    name="MortgageCalculator",
    description="Calculates monthly mortgage payment"
)


# Create the list of properly wrapped Tool instances
tools = [serpapi_tool, currency_tool, repl_tool, mortgage_tool]

react_prompt = """
You are a helpful AI agent for real estate and mortgage planning.

You can use these tools:
- PropertySearch: to find properties by location, price, or type.
- MortgageCalculator: to compute monthly payments, interest, or loan details.
- CurrencyConverter: for currency conversion.
- MarketTrendSummarizer: to summarize real estate market trends and insights.
- PropertyAdvisor: to give recommendations (e.g., buy vs rent, compare locations).

For each request:
- **Break down the user’s query into clear subtasks, and think out loud about each step.**
- **For every subtask, select and use the appropriate tool.**
    - Start each action with:
      Thought: explain what you’re about to do.
      Action: the tool name.
      Action Input: what input you’ll give the tool.
      Observation: the tool or API’s result.
- If a tool fails to return results, try another approach or let the user know.

**When you have completed all needed tool use, always give your full, final answer in a single message
beginning with "Final answer:". This answer should:**  
- Clearly summarize your findings, calculations, or recommendations.  
- Include any essential numbers (e.g., prices, rates, payment estimate, etc).  
- Give a next-step suggestion or brief, actionable advice if appropriate.

🚩 **IMPORTANT:**  
Your very last message must always start with `Final answer:` and be the only concluding message to the user.

---
Example:

User: "What is the monthly mortgage payment for a $500,000 home with a 7% interest rate over 25 years?"

Thought: I need to use the MortgageCalculator to compute this payment.
Action: MortgageCalculator
Action Input: loan amount: 500000, interest rate: 7%, duration: 25 years
Observation: The monthly payment is $3532.16.

Final answer:  
For a $500,000 home loan at 7% over 25 years, your estimated monthly payment is **$3,532.16**.  
Consider your income, downpayment, and additional costs (taxes, insurance) before finalizing your purchase. Let me know if you need region-specific advice or want to compare rent vs buy.

"""



class State(TypedDict):
    messages : Annotated[list[AnyMessage], add_messages]

def reasoning_node(state: State):
    # LLM with bound tools to enable tool-calling
    llm_with_tools = llm.bind_tools(tools)
    messages = [{"role": "system", "content": react_prompt}] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": state["messages"] + [response]}


tool_node = ToolNode(tools = tools)


def should_continue(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "content") and "final answer:" in last_message.content.lower():
        return "end"
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    if len(state["messages"]) > 20:
        return "end"
    # Otherwise, no tool_calls, not a final answer, so end gracefully
    return "end"


builder = StateGraph(State)
builder.add_node("reason", reasoning_node)
builder.add_node("action", tool_node)
builder.set_entry_point("reason")
builder.add_conditional_edges(
    "reason",
    should_continue,
    {
        "continue": "action",
        "end": END,
    }
)
builder.add_edge("action", "reason")
estatebot_agent = builder.compile()
