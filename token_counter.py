from langchain.callbacks import get_openai_callback


def run_with_token_count(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f"Query used a total of {cb.total_tokens} tokens.\n")
    return result
