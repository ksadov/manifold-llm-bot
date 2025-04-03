from pydantic import BaseModel
from typing import Optional


DEFAULT_INSTRUCTION = "You are an expert superforecaster, familiar with the work of Tetlock and others. Make a prediction of the probability that the question will be resolved as true. You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. If for some reason you can’t answer, pick the base rate, but return a number between 0 and 1."


class MarketPrediction(BaseModel):
    reasoning: str
    answer: float

    def toDict(self):
        return {"reasoning": self.reasoning, "answer": self.answer}


def format_prompt(
    scratchpad_template: Optional[str],
    question: str,
    description: str,
    creatorUsername: str,
    comments: list[dict],
    current_date: str,
) -> str:
    if scratchpad_template is not None:
        template_instruction = f"Fill in the double-bracketed sections of the template according to the instructions, using relevant information from the web if needed. Then return the filled-in reasoning template as well as your final answer.\n\n{scratchpad_template}\n\n"
    else:
        template_instruction = ""

    return f"{template_instruction}Question: {question}\nDescription: {description}\nCreator Username: {creatorUsername}\nComments: {comments}\nCurrent Date: {current_date}"
