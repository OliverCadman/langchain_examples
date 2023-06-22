import re
import os

from langchain import PromptTemplate
from langchain.chains import TransformChain, LLMChain, SequentialChain
from langchain.chat_models import ChatOpenAI
from token_counter import run_with_token_count


def transform_text(input: str) -> {}:
    text = input["text"]

    text = re.sub(r"(\r\n|\r|\n)", r" ", text)
    text = re.sub(r"[ \t]+", " ", text)

    return {
        "output_text": text
    }


clean_extra_spaces_chain = TransformChain(
    input_variables=["text"], output_variables=["output_text"], transform=transform_text)


cleaned_text = clean_extra_spaces_chain.run(
    "A random       piece of text with \n\n\n\n lots of \n\n spaces.        Another \n one here too."
)

template = """
    Paraphrase this text:
    
        {output_text}
    
    In the style of a {style}.
    
    Paraphrase:
"""

prompt = PromptTemplate(
    input_variables=["style", "output_text"], template=template
)

llm = ChatOpenAI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"), temperature=1, model="gpt-3.5-turbo")

style_paraphrase_chain = LLMChain(llm=llm, prompt=prompt, output_key="final_output")

sequential_chain = SequentialChain(
    chains=[clean_extra_spaces_chain, style_paraphrase_chain],
    input_variables=["text", "style"], output_variables=["final_output"]
)

input_text = """
    Chains allow us to              combine
    
    
    
    multiple components together to create a single, coherent application.
    For example, we can create a chain that takes a user input, format it with a PromptTemplate,
    
    then        pass the formatted
    response
    to an LLM. We can build more complex chains by combining    multiple    chains  together,
    or by combining chains with other components.
"""

if __name__ == "__main__":

    print("Generating response...\n")

    result = run_with_token_count(
        sequential_chain,
        {
            "text": input_text,
            "style": "A 90s rapper."
        }
    )

    print(result)
