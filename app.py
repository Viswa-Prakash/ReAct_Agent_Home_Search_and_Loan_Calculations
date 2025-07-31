import streamlit as st
from langchain_core.messages import HumanMessage
from ReAct_Agent import estatebot_agent

st.set_page_config(page_title="ReAct Agent for Home Search and Loan Calculations", page_icon=":house:")

st.title("ReAct Agent for Home Search and Loan Calculations")


st.markdown("""
            I’m considering a $600,000 home in Austin. 
            How have Austin property prices changed in the last 2 years? 
            If I put 15% down at 5% interest for 30 years, what’s my monthly payment?
            """)

with st.form("user_form"):
    user_query = st.text_area("Enter your query here :", height=60)
    submitted = st.form_submit_button("Ask Agent")

if submitted and user_query.strip():
    with st.spinner("Agent is analyzing..."):
        output = estatebot_agent.invoke({"messages": [HumanMessage(content=user_query)]})
        # Show **only the last agent message** (the Final Answer)
        final_message = None
        for msg in reversed(output["messages"]):
            content = getattr(msg, "content", "")
            if "final answer" in content.lower():
                final_message = content
                break
        if not final_message:
            # fallback: just show last assistant/system message
            last = output["messages"][-1]
            final_message = getattr(last, "content", str(last))
        st.markdown("**Here’s a clear summary of your requests and answers:**\n\n" + final_message)
