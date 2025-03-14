import dspy
import json
import datetime

from src.agent import make_search_tools
from src.tools.search import init_search


class GetSources(dspy.Signature):
    """Search the web and retrieve HTML content relevant to making a prediction on the given prediction market question."""

    question: str = dspy.InputField()
    description: str = dspy.InputField()
    creatorUsername: str = dspy.InputField()
    comments: list[dict] = dspy.InputField()
    current_date: str = dspy.InputField()
    answer: list[str] = dspy.OutputField()


class FillInScratchPad(dspy.Signature):
    """Fill in the double-bracketed sections of the template according to the instructions, using relevant information from the sources. Then return the filled-in template as well as your final answer."""

    template: str = dspy.InputField()
    sources: list[str] = dspy.InputField()
    filled_in: str = dspy.OutputField()
    answer: float = dspy.OutputField()


class PredictWithScratchpad(dspy.Module):
    def __init__(self, search_tools: list, template: str):
        super().__init__()
        self.template = template
        self.get_sources = dspy.ReAct(GetSources, tools=search_tools)
        self.fill_in_scratch_pad = dspy.Predict(FillInScratchPad)

    def forward(
        self,
        question: str,
        description: str,
        creatorUsername: str,
        comments: list[dict],
        current_date: str,
    ) -> dict:
        sources = self.get_sources(
            question=question,
            description=description,
            creatorUsername=creatorUsername,
            comments=comments,
            current_date=current_date,
        )
        filled_in = self.fill_in_scratch_pad(template=self.template, sources=sources)
        return filled_in


def test():
    config_path = "config/bot/basic.json"
    template_path = "src/halawi_scratchpad.txt"
    with open(config_path) as f:
        config = json.load(f)
    search = init_search(config_path, None)
    search_tools = make_search_tools(search, config["unified_web_search"])
    with open(template_path) as f:
        template = f.read()
    with open(config["llm_config_path"]) as f:
        llm_config = json.load(f)
    lm = dspy.LM(
        f'openai/{llm_config["model"]}',
        api_key=llm_config["api_key"],
        api_base=llm_config["api_base"],
        **llm_config["prompt_params"],
    )
    dspy.configure(lm=lm)
    module = PredictWithScratchpad(search_tools, template)
    question = "Will Manifold Markets shut down before 2030?"
    description = ""
    creatorUsername = "user1"
    comments = []
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    filled_in = module(
        question=question,
        description=description,
        creatorUsername=creatorUsername,
        comments=comments,
        current_date=current_date,
    )
    print(filled_in)


def main():
    test()


if __name__ == "__main__":
    main()
