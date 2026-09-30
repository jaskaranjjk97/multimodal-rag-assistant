from evaluation.deepeval_runner import DeepEvalRunner


def main():
    runner = DeepEvalRunner()

    runner.evaluate_dataset()


if __name__ == "__main__":
    main()