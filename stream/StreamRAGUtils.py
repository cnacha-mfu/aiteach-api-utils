import threading
import os
from queue import Queue, Empty
from flask import Flask, request, jsonify, Response, stream_with_context

from langchain_core.prompts import PromptTemplate 
from langchain.schema.output import LLMResult
from langchain.chains import LLMChain
from langchain_community.vectorstores import Milvus
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings.google_palm import GooglePalmEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_core.callbacks import (
    CallbackManager, 
    StreamingStdOutCallbackHandler,
    BaseCallbackHandler
)

BUFFER_SIZE = 1

milavus_connection_args = { "uri": os.getenv("MILVUS_HOST", ""), "token": os.getenv("MILVUS_API_KEY", "") }
print("milavus_connection_args: ", milavus_connection_args)

#class that takes 2 input course and input_query
class TokenStreamer:

    def __init__(self, course: str, input_query: str, token_created_callback=None):
        self.course = course
        self.input_query = input_query
        self.token_queue = Queue()
        self.token_created_callback = token_created_callback
        self.token_buffer = []

    class LLMTokenQueueHandler(BaseCallbackHandler): 
        """
        This is to change the behavior of LLMChain to 
        store the outputted tokens to a queue
        """ 
        def __init__(self, token_streamer_instance):
            self.token_streamer_instance = token_streamer_instance
        def on_llm_new_token(
            self, 
            token: str, 
            **kwargs
            ) -> None:    
            self.token_streamer_instance.token_queue.put({"type": "token", "value": token})

        def on_llm_end(
            self, 
            response: LLMResult, 
            **kwargs
            ) -> None:
            self.token_streamer_instance.token_queue.put({'type': 'end'})

    def generate_text_response(self):
        """
        Generate text response from LLM
        note that we are not streaming from this 
        function but from the stream_tokens() function
        """

        ### connect to vectordb
        print("connecting to Milvus")
        print("***query is: ", self.input_query)
        print("***course is: ", self.course)
        print("***milavus_connection_args is: ", milavus_connection_args) 

        #embedding = GooglePalmEmbeddings(model="models/text-embedding-004")
        embedding = VertexAIEmbeddings(model_name="text-embedding-004")

        # create a Milvus client
        vectordb = Milvus(
                          embedding_function=embedding,
                          connection_args=milavus_connection_args,
                          collection_name=self.course,
                         )
        
        print(vectordb.similarity_search(self.input_query))

        model = ChatGoogleGenerativeAI(model=os.getenv("MODEL_NAME", "gpt-4"),
                             temperature=0.1,
                             convert_system_message_to_human=True,
                             stream=True,
                             callback_manager=CallbackManager([TokenStreamer.LLMTokenQueueHandler(self)]))

        ### search for similar documents
        retriever = vectordb.as_retriever(search_kwargs={"k": 2})

        ### create RAG agent
        template = """
                    You are a helpful AI assistant.
                    Given the context below, please answer the question.
                    context: {context}
                    input: {input}
                    answer:
                    """
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        combine_docs_chain = create_stuff_documents_chain(model, QA_CHAIN_PROMPT)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        retrieval_chain = retrieval_chain.pick("answer")
        #this streaming call triggers the process to 
        #store answer tokens to queue
        # for chunk_response in llm_chain.stream(
        for chunk_response in retrieval_chain.stream(
            {'input': self.input_query}
        ):
            print(chunk_response)


    def stream_tokens(self):  
        """Generator function to stream tokens."""  
        while True:  
            # Wait for a token to be available in the queue and retrieve it  
            token_dict = self.token_queue.get()
            print("token_dict: ", token_dict)

            if token_dict["type"] == "token":
                self.token_buffer.append(token_dict.get("value"))
           
            if len(self.token_buffer) > BUFFER_SIZE and self.token_created_callback:
                # join all token in buffer to form a sentence
                sentence = "".join(self.token_buffer)
                self.token_created_callback(sentence)
                self.token_buffer = []

            if token_dict["type"] == "token":
                # encode str as byte  
                yield token_dict['value'].encode('utf-8')

            #we need to implement when streaming ends
            #with the 'end' token, then break out of loop
            elif token_dict["type"] == "end":
                ## update what is left and break
                sentence = "".join(self.token_buffer)
                self.token_created_callback(sentence)
                self.token_buffer = []
                break

    def run(self):
        t1 = threading.Thread(
            target=self.generate_text_response
        )
        t1.start()
