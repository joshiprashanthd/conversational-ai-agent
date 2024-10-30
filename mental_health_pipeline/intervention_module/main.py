import json
from dotenv import load_dotenv


from langchain_community.document_loaders import JSONLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

llm = ChatGroq(model="llama3-8b-8192", stop_sequences=[])

loader = JSONLoader(
    file_path="../../data/intervention-migrate.json",
    jq_schema=".[]",
    text_content=False,
)

docs = loader.load()

# json_splitter = RecursiveJsonSplitter(max_size=1000)
# all_splits = json_splitter.split(docs)
# print(all_splits[0])

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})


def format_docs(docs):
    res = []
    template = """Activity: {activity}
Type: {type}
Reasoning: {reasoning}
How to do it: {howto}
Affecting Params:
{affecting_params}
"""
    affect_params_template = """Parameter Name: {param}
Effectiveness: {effectiveness}
"""

    for doc in docs:
        data = json.loads(doc.page_content)

        tmp = []

        for obj in data["affecting_params"]:
            tmp.append(
                affect_params_template.format(
                    param=obj["name"], effectiveness=obj["effectiveness"]
                ),
            )

        res.append(
            template.format(
                activity=data["activity"],
                type=data["type"],
                reasoning=data["reasoning"],
                howto=data["howto"],
                affecting_params="\n".join(tmp),
            )
        )

    return "\n\n".join(res)


def print_prompt(prompt):
    # print("PROMPT:: ", prompt)
    return prompt


template = """Use the following data to pick top 3 activities based on patient mental state. The data consist of various activities that could be suggested to a patient. The activities affects several parameters of mental state. For example, "Prasarita Padottanasana" activity has "low" effectiveness on "ease of falling asleep" parameter, so it is likely to suggest this activity when patient is experiencing difficulty in falling asleep. Try to choose activities that are highly effective for the given mental state.

Output Instructions:
    - Strictly choose activities from the data provided to you.
    - Do not repeat answer in your response.
    - Enclose all the comma separated activities you generate with left bracket [ and right bracket ]

{context}

Mental State: {mental_state}

Activity Names: """
custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "mental_state": RunnablePassthrough()}
    | custom_rag_prompt
    | print_prompt
    | llm
    | StrOutputParser()
)

print(
    rag_chain.invoke(
        "This patient is experiencing significant sleep disturbances that are negatively impacting their overall well-being. The constant hum in their head and very fragmented sleep with frequent awakenings indicate poor sleep continuity. Additionally, feeling cold despite the heat being on may be further disrupting their sleep. Prior to sleep, anxious and worried thoughts, such as racing thoughts and concerns about the past and future, may be contributing to their sleep difficulties. Upon waking, they report feeling extremely tired and disconnected, with feelings of heaviness and pointlessness, indicating poor daytime functioning. The lack of vivid dreams suggests that the source of their sleep disturbances may be primarily environmental or related to cognitive-emotional issues prior to sleep, rather than related to nightmares or distressing dream content."
    )
)
