import dspy
from logging import Logger
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import datetime
from src.logging import create_logger
import os

from src.tools.search import init_search
from src.agent.dspy_agents import init_dspy, stringify_for_logging
from src.agent.openai_agent import init_openai
from src.agent.google_agent import init_google


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
    if config["agent_type"] == "dspy":
        predict_market = init_dspy(
            llm_config,
            config["dspy_program_path"],
            search,
            config["unified_web_search"],
            config["use_python_interpreter"],
            scratchpad_template_path,
            logger,
        )
    elif config["agent_type"] == "openai":
        predict_market = init_openai(
            llm_config,
            search,
            logger,
            config["unified_web_search"],
            config["use_python_interpreter"],
            scratchpad_template_path,
        )
    elif config["agent_type"] == "google":
        raise ValueError("Google agent is broken, do not use")
        predict_market = init_google(
            llm_config,
            search,
            logger,
            config["unified_web_search"],
            config["use_python_interpreter"],
            scratchpad_template_path,
        )
    else:
        raise ValueError(f"Invalid agent type: {config['agent_type']}")
    return (
        predict_market,
        logger,
        evalfile_name,
        cutoff_date,
        config["market_filters"]["exclude_groups"],
    )
