import os
from langchain.chains import LLMMathChain
from langchain.chat_models import ChatOpenAI

from token_counter import run_with_token_count

prompt = "What is 12 raised to the 14th power?"

llm = ChatOpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), model="gpt-3.5-turbo", temperature=0)

chain = LLMMathChain.from_llm(llm=llm, verbose=True)

if __name__ == "__main__":
    print("Generating response...\n")
    result = run_with_token_count(chain, prompt)

    print(result)
