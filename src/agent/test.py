from src.agent import init_pipeline
import datetime


def test():
    config_path = "config/bot/test.json"
    predict_market, _, _, _, _ = init_pipeline(
        config_path,
        "INFO",
        "deploy",
    )
    question = "Will Manifold Markets shut down this year?"
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
