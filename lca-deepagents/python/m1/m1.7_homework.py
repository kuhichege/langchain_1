# python/m1/m1.7_homework_filled.py
"""Reference copy of m1.7_homework.py with TODOs 1 and 2 filled in so you
can run it end to end and see what "done" looks like. This is just one
possible answer, so yours might be different. Explore!"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver

from models import model

agent = create_deep_agent(
    model=model,
    checkpointer=MemorySaver(),
)

# TODO 1
thread_a = {"configurable": {"thread_id": "thread-a"}}
thread_b = {"configurable": {"thread_id": "thread-b"}}

FACT_QUESTION = "What is my best friend's favorite food?"


# TODO 2
def run_scenario():
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Remember that my best friend's favorite food is ugali."}]},
        config=thread_a,
    )
    print("Thread A, turn 1:")
    print(result["messages"][-1].text)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": FACT_QUESTION}]},
        config=thread_a,
    )
    print(result["messages"][-1].text)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": FACT_QUESTION}]},
        config=thread_b,
    )
    print(result["messages"][-1].text)

    # A brand-new agent with its own fresh MemorySaver has no history
    fresh_agent = create_deep_agent(model=model, checkpointer=MemorySaver())
    result = fresh_agent.invoke(
        {"messages": [{"role": "user", "content": FACT_QUESTION}]},
        config=thread_a,
    )
    print("\nFresh agent, thread_a's thread_id (should NOT know):")
    print(result["messages"][-1].text)

run_scenario()
