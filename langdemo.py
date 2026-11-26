from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool


# =========================================================
# 1. OLLAMA LLM
# =========================================================

llm = ChatOllama(
    model=" llama3.1:8b ",
    temperature=0
)


# =========================================================
# 2. TOOLS
# =========================================================

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def get_student_marks(name: str) -> str:
    """Get marks of a student."""

    students = {
        "ram": 85,
        "john": 72,
        "alice": 91
    }

    marks = students.get(name.lower())

    if marks is None:
        return "Student not found"

    return f"{name} scored {marks} marks"


# Put all three tools into one list
tools = [
    add,
    multiply,
    get_student_marks
]


# =========================================================
# 3. CREATE AGENT
# =========================================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
    You are a helpful AI assistant.

    You have access to three tools:

    1. add - adds two numbers
    2. multiply - multiplies two numbers
    3. get_student_marks - gets student marks

    Always use the appropriate tool when required.
    """
)


# =========================================================
# 4. FUNCTION TO RUN AGENT
# =========================================================

def ask_agent(question):

    response = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    })

    print("\nUSER:")
    print(question)

    print("\nAGENT:")
    print(response["messages"][-1].content)

    print("-" * 50)


# =========================================================
# 5. TEST ALL THREE TOOLS
# =========================================================

ask_agent("What is 25 plus 15?")

ask_agent("What is 25 multiplied by 4?")

ask_agent("What are Ram's marks?")