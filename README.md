This repository contains code and instructions for setting up an LLM agent based trading bot for https://manifold.markets/. This agent is compatible with most LLM backends and comes with customizable prompts and scripts for backtesting on historical Manifold data.

# Setup
1. Clone this repo and set up a virtual environment, then install requirements with `pip install -r requirements.txt`. I've tested on Mac and Linux with python 3.11, you're on your own if it breaks for Windows.
2. Specify an LLM config. LLM configuration is compatible with any provider that uses an [OpenAI-compatible](https://github.com/openai/openai-openapi) endpoint, which includes OpenAI, Anthropic, Together AI, [llama.cpp's default server](https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md) and more. Consult `config/llm/gpt-4o-mini-example.json` for reference. Knowledge cutoff date is only required for backtesting on historical data, you can trade without it.
3. Set up a [Google Custom Search engine](https://developers.google.com/custom-search/v1/introduction) and obtain a your Programmable Search Engine identifier as well as a Google API key.
4. Obtain a [Manifold Markets API key](https://docs.manifold.markets/api#authentication).
5. Use your Programmable Search Engine identifier, Google API key and Manifold Markets API key to create a secrets config (see `config/secrets/secrets-example.json` for reference)
6. Edit your bot config (see `config/bot/basic.json` for reference) to point at your LLM config and your secrets config.

# Trade
Once you have a bot config set up, run `python -m src.scripts.trade my/config/path` to start trading. If you just want to test the bot and aren't ready for it to interact with Manifold for real just yet, set the config value `dry_run` to `true` and `comment_with_reasoning` to false.

# Backtest
1. Download [bets, markets and comments dumps](https://docs.manifold.markets/api#trade-history-dumps). If you'd like you can inspect the contents of each file with `src.scripts.inspect_data_dump path/to/json`.
2. Run `python -m src.scripts.make_dataset --markets_filepath some/path --trades_filepath some/other/path --comments_filepath you/get/the/idea` to combine the data into a parquet file.
3. Run `python -m src.scripts.make_data_split` in order to create test, val and train parquet files.
4. You can try to run `python -m src.scripts.evaluate --config_path config/bot/my_config.json --max_examples $SOME_REASONABLE_NUMBER --num_workers $SOME_OTHER_NUMBER` to use DSPy's built-in evaluation utility, but if `$SOME_REASONABLE_NUMBER > 10` and `$SOME_OTHER_NUMBER > 1` it may hang indefinitely. I recommend instead using `python -m src.scripts.dirty_evaluate --config_path config/bot/my_config.json --max_examples $SOME_REASONABLE_NUMBER  --num_threads $SOME_OTHER_NUMBER`. The latter script also lets you specify a `--timeout` value in seconds, which gracefully fails examples which take longer than that value to complete.

# Optimize
`python -m src.scripts.optimize` will let you run optimization using DSPY's implementation of MIRPOv2 or COPRO depending on flags. But it is again likely to hang, so IMO you're better off editing the programs by hand and then linking them in the bot config: see `dspy_programs/halawi_zero_shot.json` for an example.

# Credits
The prompts in `dspy_programs/halawi_zero_shot.json` and  `config/templates/halawi_scratchpad.txt` were adapted from https://arxiv.org/html/2402.18563v1.
