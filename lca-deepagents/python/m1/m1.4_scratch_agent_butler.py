from deepagents import create_deep_agent

from models import model

SYSTEM_PROMPT = (
    "You are a salty pirate captain with decades at sea. Talk in 'arr' and nautical slang. "
    "Call the user 'matey' and frame your answers as if charting a course."
)

agent = create_deep_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    name="butler_agent",
)

result = agent.invoke({"messages": [{"role": "user", "content": "What is an LLM?"}]})

print(result["messages"][-1].text)
