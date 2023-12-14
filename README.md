# LangServeDemo
A demonstration of using LangServe to create an API from a LCEL Rag Chain!

## Step 1: Build an Index

Using the Notebook found [here](https://github.com/AI-Maker-Space/LangServeDemo/blob/main/create_index.ipynb) - we created, and then saved, a FAISS-backed VectorStore containing information from the LangServe repository.

## Step 2: Build an LCEL Chain

Within the `server.py` found [here](https://github.com/AI-Maker-Space/LangServeDemo/blob/main/app/server.py), we can create our chain. 

We leverage our pre-created index and some Hugging Face Inference endpoints (hosting Mistral-7B-Instruct-v0.1, and WhereIsAI/UAE-Large-V1 embeddings) through LangChain - and then create a simple RAG chain using LCEL.

```python
hf_llm = HuggingFaceEndpoint(
    endpoint_url="<<YOUR URL HERE>>",
    huggingfacehub_api_token=os.environ["HF_TOKEN"],
    task="text-generation",
)

embeddings_model = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.environ["HF_TOKEN"],
    api_url="<<YOUR URL HERE>>",
)

faiss_index = FAISS.load_local("langserve_index", embeddings_model)
retriever = faiss_index.as_retriever()

prompt_template = """\
Use the provided context to answer the user's question. If you don't know the answer, say you don't know.

Context:
{context}

Question:
{question}"""

rag_prompt = ChatPromptTemplate.from_template(prompt_template)

entry_point_chain = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)
rag_chain = entry_point_chain | rag_prompt | hf_llm | StrOutputParser()
```

## Step 3: Add the Custom Route

Now we can map our chain to its own custom route using:

```python
add_routes(app, rag_chain, path="/rag")
```

## Step 4: Serve

All that's left to do is:

`langchain serve`!

Head on over to `localhost:8000/rag/playground` to get experimenting with your chain!


