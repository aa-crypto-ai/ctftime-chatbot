from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference

from llm.models import ChatModel

def get_precision(prompt: str, response: str, chat_model: ChatModel, docs: list):

    evaluator_llm = LangchainLLMWrapper(chat_model.chat_model)
    context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)

    docs_content = [doc.page_content for doc, sim_score in docs]

    # max 20 tries, OpenRouter API could be unstable
    for _ in range(20):
        try:
            sample = SingleTurnSample(
                user_input=prompt,
                response=response,
                retrieved_contexts=docs_content,
            )

            # note it could return 0.9999999999 == 1 / (1+1e-10) when theoretically it should be 1
            # because of this line adding 1e-10 to the denominator for preventing 0 denom when there are no valid verdicts
            # https://github.com/xi-zhou/ragas/blob/ae48b4e837dd57964356a4c977724c15f93936c4/src/ragas/metrics/_context_precision.py#L132
            context_precision = context_precision.single_turn_score(sample)   # there is an async version
            return context_precision

        except Exception as e:
            print('eval', str(e))
            # do logging
            continue

    return None
