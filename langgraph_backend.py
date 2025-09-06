from langgraph.graph import StateGraph,END,START
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from typing import TypedDict,Annotated
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage
import os 
load_dotenv()


class chatbot(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


checkpointer = MemorySaver()
graph = StateGraph(chatbot)

llm = ChatOpenAI(model='gpt-3.5-turbo',api_key=os.getenv('OPENAI_API_KEY'))

def message_reply(state:chatbot):
    ai_message = llm.invoke(state['messages'])
    return {'messages':[ai_message]}

graph.add_node('message_reply',message_reply)

graph.add_edge(START,'message_reply')
graph.add_edge('message_reply',END)


workflow = graph.compile(checkpointer=checkpointer)


config = {'configurable':{'thread_id':'thread_2'}}

persona = SystemMessage(content=(
      "You are the best data scientist in the world. "
    "You answer every question concisely and in a mentor-like way. "
    "Always focus on practical insights, clarity, and correctness."
))

# Add the persona as the first message
workflow.invoke({"messages": [persona]}, config=config)
# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ["quit", "exit"]:
#         break
#     result = workflow.invoke(
#         {"messages": [HumanMessage(content=user_input)]},
#         config=config
#     )
#     print("AI:", result['messages'][-1].content)

print(workflow.get_state(config=config))
