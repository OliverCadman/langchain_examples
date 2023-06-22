from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

import os

haiku_template = """
    I want you to write me a haiku about an {object}.
"""

prompt_template = PromptTemplate(
    input_variables=["object"], template=haiku_template
)

# MAKE SURE YOU HAVE AN OPENAI API KEY IN YOUR ENVIRONMENT.
llm = ChatOpenAI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"), temperature="1", model_name="gpt-3.5-turbo")

chain = LLMChain(llm=llm, prompt=prompt_template)

if  __name__ == "__main__":
    dynamic_object = "orange"

    print("Generating haiku...\n")
    result = chain.run(object=dynamic_object)
    print(result)
