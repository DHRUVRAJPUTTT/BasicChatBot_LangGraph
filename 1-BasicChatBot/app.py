import os
import datetime
import math
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, Request
import json
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")


@tool
def get_current_time() -> str:
    """Get the current date and time. Use this when the user asks about the current time or date."""
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M %p")


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Use this when the user asks for a calculation.
    Examples: '2 + 2', '15 * 7', 'sqrt(144)', '2**10'
    """
    try:
        
        allowed = {
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "pi": math.pi,
            "e": math.e, "abs": abs, "round": round, "pow": pow,
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def search_knowledge(query: str) -> str:
    """Search the knowledge base for information on a topic.
    Use this when the user asks factual questions about science, history, geography, etc.
    """
    knowledge = {
        "mitochondria": "The mitochondria is the powerhouse of the cell. It generates most of the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy.",
        "python": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with cyclic graph support for complex agent architectures.",
        "india": "India is a country in South Asia. Its capital is New Delhi. The current Prime Minister is Narendra Modi and the President is Droupadi Murmu.",
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return f"No specific entry found for '{query}'. The LLM will use its general knowledge to answer."


# Tavily web search for real-time information
web_search = TavilySearch(
    max_results=3,
    topic="general",
    search_depth="basic",
)

tools = [get_current_time, calculate, search_knowledge, web_search]

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. You have access to the following tools:\n"
    "- get_current_time: Returns the current date and time.\n"
    "- calculate: Evaluates a mathematical expression (e.g. '2+2', 'sqrt(144)').\n"
    "- search_knowledge: Searches a small knowledge base for factual topics.\n"
    "- tavily_search: Searches the web for real-time, up-to-date information.\n\n"
    "IMPORTANT: For questions about current events, news, prices, weather, sports scores, "
    "or anything that requires up-to-date information, you MUST use the tavily_search tool. "
    "Do NOT rely on your training data for current facts.\n"
    "For general conversation or simple greetings, answer directly without tools."
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
)
llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    """Node function: invokes the LLM (with tools) on current messages."""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Build the graph with tool-calling loop
graph_builder = StateGraph(State)
graph_builder.add_node("llmchatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))

# Edges: START -> LLM -> (conditional: tool call or END) -> tools -> LLM
graph_builder.add_edge(START, "llmchatbot")
graph_builder.add_conditional_edges("llmchatbot", tools_condition)
graph_builder.add_edge("tools", "llmchatbot")

graph = graph_builder.compile()


#FastAPI App
app = FastAPI(title="LangGraph Chatbot", version="2.0.0")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint for HF Spaces."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the chat UI."""
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/playground", response_class=HTMLResponse)
async def serve_playground():
    """Serve the Developer Playground UI."""
    html_path = os.path.join(os.path.dirname(__file__), "templates", "playground.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/chat")
async def chat_endpoint(request: Request):
    """
    Accept a user message, invoke the LangGraph chatbot, return the AI reply.
    Now also returns tool call traces for the playground UI.

    Expects JSON: {"message": "Hello!"}
    Returns JSON: {"reply": "...", "tool_calls": [...]}
    """
    try:
        body = await request.json()
        user_message = body.get("message", "").strip()

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={"error": "Message cannot be empty."},
            )

        # Build messages with system prompt + user message
        input_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            HumanMessage(content=user_message),
        ]

        async def event_generator():
            try:
                async for event in graph.astream_events({"messages": input_messages}, version="v2"):
                    kind = event["event"]
                    if kind == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        if chunk.content and isinstance(chunk.content, str):
                            yield f"data: {json.dumps({'type': 'message_chunk', 'content': chunk.content})}\n\n"
                    elif kind == "on_tool_start":
                        yield f"data: {json.dumps({'type': 'tool_call', 'tool_name': event['name'], 'arguments': event['data'].get('input')})}\n\n"
                    elif kind == "on_tool_end":
                        out = event['data'].get('output')
                        if hasattr(out, "content"):
                            out = out.content
                        yield f"data: {json.dumps({'type': 'tool_result', 'tool_name': event['name'], 'result': out})}\n\n"
                
                yield "data: [DONE]\n\n"
            except Exception as graph_err:
                err_str = str(graph_err)
                if "tool_use_failed" in err_str or "tool_call" in err_str.lower():
                    # Fallback stream
                    try:
                        async for chunk in llm.astream(input_messages):
                            if chunk.content and isinstance(chunk.content, str):
                                yield f"data: {json.dumps({'type': 'message_chunk', 'content': chunk.content})}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as fallback_err:
                        yield f"data: {json.dumps({'type': 'error', 'content': str(fallback_err)})}\n\n"
                        yield "data: [DONE]\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'error', 'content': str(graph_err)})}\n\n"
                    yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred: {str(e)}"},
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
