#from langchain_openai import ChatOpenAI

from openai import AzureOpenAI

# -----------------------------
# Hardcoded Azure OpenAI credentials
# -----------------------------
AZURE_OPENAI_ENDPOINT= "https://bts-poc-openai.openai.azure.com/"
AZURE_OPENAI_API_KEY= "B5fT1mCS7uCB4L3fBlKNAQ9UTLN5E9v8dUflNyD9VmSU9Cblie6LJQQJ99BFAC77bzfXJ3w3AAABACOGXToO"
AZURE_OPENAI_API_VERSION="2025-01-01-preview"
AZURE_OPENAI_DEPLOYMENT="gpt-4.1-mini"
# -----------------------------
# Initialize Azure OpenAI client
# -----------------------------
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# SEARCH (RETRIEVAL)
def search(vectordb, query, k=3):
    return vectordb.similarity_search(query, k=k)

# ASK LLM (ANSWER GENERATION)
def ask_llm(context_docs, question):
    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    system_prompt = """
You are a helpful assistant. Use ONLY the following context to answer.
"""
    user_prompt = f"""
CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:
(Follow all rules. If not found in context, say exactly:
"I don't know based on the PDF.")
"""

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        temperature=0,
        max_tokens=500,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    )

    # Correct for Azure SDK
    return response.choices[0].message.content
