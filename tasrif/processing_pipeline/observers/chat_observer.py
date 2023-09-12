"""Module that defines the Chat class
"""
from tasrif.processing_pipeline.observers.functional_observer import FunctionalObserver
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
import os


class ChatObserver(FunctionalObserver):
    """Chat class to to chat with a dataframe using an LLM"""

    def __init__(self, query=""):
        """
        The constructor of the Chat class will provide options to configure the
        operation. The chat is invoked via the observe method and the data to be
        chatted with is passed to the observe method.

        Args:
            query (String):
                Query to be asked to LLM
        """
        # self._logging_methods = []
        self._query = query

    def _observe(self, operator, *data_frames):
        """
        Observe the passed data using the processing configuration specified
        in the constructor

        Args:
            operator (ProcessingOperator):
                Processing operator which is observed
            *data_frames (list of pd.DataFrame):
                Variable number of pandas dataframes to be observed
        """
        template = """
        You are an expert in personal health. Given the following context containing health data of a user, give a short response that best answers the users question.
        Context: {data_frames}
        Question: {query}
        """

        # API key for OpenAI
        os.environ["OPENAI_API_KEY"] = "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"

        if os.environ["OPEN_API_KEY"] == "":
            raise Exception("OPEN API KEY required")

        prompt = PromptTemplate(template=template, input_variables=[
                                "data_frames", "query"])
        final_prompt = prompt.format(
            data_frames=data_frames, query=self._query)
        print(final_prompt)

        llm = OpenAI()
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        print(llm_chain.predict(query=self._query, data_frames=data_frames))

    def observe(self, operator, *data_frames):
        """
        Function that performs checks on operator and data frame before observation
        This observation is only performed on non-infrastructure operators

        Args:
            operator (ProcessingOperator):
                Processing operator which is observed
            *data_frames (list of pd.DataFrame):
                Variable number of pandas dataframes to be observed
        """

        if operator.is_functional():
            self._observe(operator, *data_frames)
