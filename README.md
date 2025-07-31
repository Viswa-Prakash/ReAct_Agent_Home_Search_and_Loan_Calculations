# ReAct Agent for Home Search and Loan Calculations

**HomeSmart** is an intelligent, LangGraph-powered ReAct agent designed to assist users with real estate exploration and mortgage planning. It combines LLM reasoning with powerful tools like property search, mortgage calculators, and real-time currency conversion to offer a one-stop smart home-buying assistant.

---

##  Features

-  **Property Search**: Find real estate listings based on location, price range, and preferences using SerpAPI.
-  **Mortgage Calculator**: Compute monthly payments for various loan terms and interest rates.
-  **Currency Converter**: Convert property prices across currencies using Alpha Vantage.
-  **Market Trend Summarizer**: Summarize real estate trends from news and blogs.
-  **AI Advisor**: Get intelligent recommendations on buying, renting, or comparing property options.

---

##  Built With

- **[LangChain](https://github.com/langchain-ai/langchain)** – Framework for building LLM-powered apps
- **[LangGraph](https://github.com/langchain-ai/langgraph)** – For ReAct-style stateful workflows
- **[OpenAI](https://openai.com/)** – LLM for reasoning and summarization
- **[SerpAPI](https://serpapi.com/)** – For fetching real estate listings and market trends
- **[Alpha Vantage](https://www.alphavantage.co/)** – Real-time currency conversion
- **Python** – Tool logic and API integration


---

##  Tools Used in the Agent

| Tool Name           | Description                                  |
|---------------------|----------------------------------------------|
| `PropertySearch`    | Finds listings via SerpAPI                   |
| `MortgageCalculator`| Computes monthly mortgage payment            |
| `CurrencyConverter` | Converts currency via Alpha Vantage API      |
| `MarketTrendSummarizer` | Summarizes housing market news          |
| `PropertyAdvisor`   | Uses LLM to provide buy/rent suggestions     |

---

##  Example Prompts

- *"Find a 3BHK apartment in Bangalore under ₹1.5Cr."*
- *"What will be my EMI if I borrow $250,000 for 20 years at 6.5% interest?"*
- *"Convert $300,000 to INR using the current rate."*
- *"Summarize current real estate trends in Mumbai."*
- *"Should I rent or buy a house in San Francisco?"*

---

##  Getting Started

```bash
1. Clone the Repository
```bash
git clone https://github.com/Viswa-Prakash/ReAct_Agent_Home_Search_and_Loan_Calculations.git
cd ReAct_Agent_Home_Search_and_Loan_Calculations


2. Set Up Environment
```bash
conda create -p venv python=3.13 -y
conda activate venv
pip install -r requirements.txt

3.  Add .env File
```bash
- SERPAPI_API_KEY=your_key
- ALPHA_VANTAGE_API_KEY=your_key
- OPENAI_API_KEY=your_key

4. Run the Agent
```bash
streamlit run app.py

---

##  Example Interaction

**User Query:**
> "I’m planning to buy a house in New York for $650,000. What would my monthly mortgage be for a 20-year loan at 6.5% interest? Also, can you convert that amount to INR?"

**Agent Workflow:**
Thought: The user wants to know the monthly mortgage for a $650,000 loan and also wants the amount in INR. I will use the MortgageCalculator and CurrencyConverter tools.

Action: MortgageCalculator
Action Input: 650000, 6.5, 20
Observation: Monthly Payment: $4,833.02

Action: CurrencyConverter
Action Input: 650000, USD, INR
Observation: $650000 = ₹54,37,500.00

Final Answer:
For a $650,000 property over 20 years at 6.5% interest, your estimated monthly mortgage payment would be $4,833.02. The total amount in Indian Rupees is approximately ₹54,37,500.00.


This example showcases how the agent chains tool usage to arrive at an insightful, personalized response using ReAct logic.



