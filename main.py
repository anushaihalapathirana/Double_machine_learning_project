from src.experiment import config_from_args, parse_args, run_experiment


def main():
    run_experiment(config_from_args(parse_args()))


if __name__ == "__main__":
    main()
