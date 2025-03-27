import dspy
from logging import Logger
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import datetime
from src.logging import create_logger
import os

from typing import Any, Dict, Optional

from src.tools.search import init_search
from src.agent.dspy_agents import init_dspy, stringify_for_logging


def init_pipeline(
    config_path: Path,
    log_level: str,
    mode: str,
) -> Tuple[List[dspy.Example], dspy.ReAct, Logger, Optional[str]]:
    with open(config_path) as f:
        config = json.load(f)
    llm_config_path = Path(config["llm_config_path"])
    with open(llm_config_path) as f:
        llm_config = json.load(f)
    # specified in 2021-01-01 format
    if "knowledge_cutoff" in llm_config and mode != "deploy":
        cutoff_date = datetime.datetime.strptime(
            llm_config["knowledge_cutoff"], "%Y-%m-%d"
        )
    else:
        cutoff_date = None

    logger, logfile_name = create_logger(config["name"], mode, log_level=log_level)
    if mode == "eval":
        evalfile_name = f"logs/{mode}/{logfile_name.split('.')[0]}.json"
        os.makedirs(f"logs/{mode}", exist_ok=True)
    else:
        evalfile_name = None
    logger.info(f"Config: {config_path}")
    logger.info(f"Config: {stringify_for_logging(config)}")
    search = init_search(config_path)

    scratchpad_template_path = (
        Path(config["scratchpad_template_path"])
        if "scratchpad_template_path" in config and config["scratchpad_template_path"]
        else None
    )

    # Initialize prediction function
    predict_market = init_dspy(
        llm_config,
        config["dspy_program_path"],
        search,
        config["unified_web_search"],
        config["use_python_interpreter"],
        scratchpad_template_path,
        logger,
    )
    return (
        predict_market,
        logger,
        evalfile_name,
        cutoff_date,
        config["market_filters"]["exclude_groups"],
    )


def test():
    config_path = "config/bot/scratchpad.json"
    predict_market, _, _, _, _ = init_pipeline(
        config_path,
        "INFO",
        "deploy",
    )
    question = "Will Manifold Markets shut down before 2030?"
    description = ""
    creatorUsername = "user1"
    comments = []
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    filled_in = predict_market(
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
