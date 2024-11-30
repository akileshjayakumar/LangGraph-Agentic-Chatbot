import os
import uuid
import logging
from dotenv import load_dotenv
import json
import gradio as gr
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage
from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI as Chat

from uuid import uuid4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# LangGraph setup
openai_api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-4")
temperature = float(os.getenv("OPENAI_TEMPERATURE", 0))

web_search = TavilySearchResults(max_results=2)
tools = [web_search]


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


llm = Chat(
    openai_api_key=openai_api_key,
    model=model,
    temperature=temperature
)
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(
            f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


tool_node = BasicToolNode(tools=[web_search])
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")


def chatbot(state: State):
    if not state["messages"]:
        logger.info(
            "Received an empty message list. Returning default response.")
        return {"messages": [AIMessage(content="Hello! How can I assist you today?")]}

    # Check for tool call in the last message
    last_message = state["messages"][-1]
    if not getattr(last_message, "tool_calls", None):
        logger.info(
            "No tool call in the last message. Proceeding without tool invocation.")
        response = llm.invoke(state["messages"])
    else:
        logger.info(
            "Tool call detected in the last message. Invoking tool response.")
        response = llm_with_tools.invoke(state["messages"])

    # Ensure the response is wrapped as AIMessage if it's not already
    if not isinstance(response, AIMessage):
        response = AIMessage(content=response.content)

    return {"messages": [response]}


graph = graph_builder.compile()


def gradio_chat(message, history):
    try:
        if not isinstance(message, str):
            message = str(message)

        config = {
            "configurable": {"thread_id": "1"},
            "checkpoint_id": str(uuid4()),
            "recursion_limit": 300
        }

        # Format the user message correctly as a HumanMessage
        formatted_message = [HumanMessage(content=message)]
        response = graph.invoke(
            {
                "messages": formatted_message
            },
            config=config,
            stream_mode="values"
        )

        # Extract assistant messages and ensure they are AIMessage type
        assistant_messages = [
            msg for msg in response["messages"] if isinstance(msg, AIMessage)
        ]
        last_message = assistant_messages[-1] if assistant_messages else AIMessage(
            content="No response generated.")

        logger.info("Sending response back to Gradio interface.")
        return last_message.content
    except Exception as e:
        logger.error(f"Error encountered in gradio_chat: {e}")
        return "Sorry, I encountered an error. Please try again."


with gr.Blocks(theme=gr.themes.Default()) as demo:
    chatbot = gr.ChatInterface(
        chatbot=gr.Chatbot(height=800, render=False),
        fn=gradio_chat,
        multimodal=False,
        title="LangGraph Agentic Chatbot",
        examples=[
            "What's the weather like today?",
            "Show me the Movie Trailer for Doctor Strange.",
            "Give me the latest news on the COVID-19 pandemic.",
            "What are the latest updtaes on NVIDIA's new GPU?",
        ],
    )

if __name__ == "__main__":
    demo.launch()
