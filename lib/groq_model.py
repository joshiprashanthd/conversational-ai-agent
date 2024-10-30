from groq import Groq
from groq.types.chat import ChatCompletionMessageParam
import os

from dotenv import load_dotenv

load_dotenv()


class GroqModel:
    def __init__(self, name: str, model_id: str, api_key=os.getenv("GROQ_API_KEY")):
        self.system_prompt: str = (
            'You are a helpful assistant. You do not reply with irrelevant text such as "Here is your response..." and only perform the task that is given to you.'
        )
        self.name: str = name
        self.model_id: str = model_id
        self.client = Groq(api_key=api_key)

    def completion(self, prompt: str, **kwargs):
        system_prompt: ChatCompletionMessageParam = {
            "role": "system",
            "content": self.system_prompt if len(self.system_prompt) > 0 else "",
        }
        user_prompt: ChatCompletionMessageParam = {
            "role": "user",
            "content": prompt.format(**kwargs),
        }
        chat_history = [system_prompt, user_prompt]

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=chat_history,
            max_tokens=kwargs["max_tokens"] if "max_tokens" in kwargs else 1024,
            stream=kwargs["stream"] if "stream" in kwargs else False,
            seed=91234,
        )

        return response.choices[0].message.content

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt


class Llama3_8BGroq(GroqModel):
    def __init__(self, api_key=os.getenv("GROQ_API_KEY")):
        super().__init__("Llama-3-8B-Groq", "llama3-8b-8192", api_key=api_key)


class Llama3_70BGroq(GroqModel):
    def __init__(self, api_key=os.getenv("GROQ_API_KEY")):
        super().__init__(
            name="Llama3-70B-Groq", model_id="llama3-70b-8192", api_key=api_key
        )


class Mixtral8x7BGroq(GroqModel):
    def __init__(self, api_key=os.getenv("GROQ_API_KEY")):
        super().__init__(
            name="Mixtral8x7B-Groq", model_id="mixtral-8x7b-32768", api_key=api_key
        )


class Llama3_1_405B_Reasoning(GroqModel):
    def __init__(self, api_key=os.getenv("GROQ_API_KEY")):
        super().__init__(
            name="Llama-3.1-405B-Reasoning",
            model_id="llama-3.1-405b-reasoning",
            api_key=api_key,
        )


class Llama3_1_70B_Versatile(GroqModel):
    def __init__(self, api_key=os.getenv("GROQ_API_KEY")):
        super().__init__(
            name="Llama-3.1-70B-Versatile",
            model_id="llama-3.1-70b-versatile",
            api_key=api_key,
        )


class Llama3_1_8B_Instant(GroqModel):
    def __init__(self, api_key=os.getenv("GROQ_API_KEY")):
        super().__init__(
            name="Llama-3.1-8B-Instant",
            model_id="llama-3.1-8b-instant",
            api_key=api_key,
        )
