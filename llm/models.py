from utils.langchain_adapter import ChatOpenRouter

class ChatModel:

    def __init__(self, family_name, model_name_short, display_name=None):
        self.family_name = family_name
        self.model_name_short = model_name_short
        self.model_name = f'{self.family_name}/{self.model_name_short}'
        self.display_name = display_name
        self.chat_model = ChatOpenRouter(model_name=self.model_name)

    def invoke(self, msgs):
        """ get response from the LLM inference with the list of messages (HumanMessage / AIMessage)
            note: in case of OpenRouter API stability issue, try 20 times
        """
        for _ in range(20):
            try:
                return self.chat_model.invoke(msgs)
            except:
                # do logging
                continue

    @classmethod
    def from_names(cls, family_name, model_name_short, display_name=None):
        return cls(family_name, model_name_short, display_name)

    def __repr__(self):
        if self.display_name is None:
            return f'<ChatModel> {self.model_name}'
        else:
            return f'<ChatModel> {self.display_name} - {self.model_name}'
