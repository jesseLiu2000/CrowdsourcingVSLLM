import argparse
from src.persuasiveness import PersuasivenessDealing
from src.imdb import ImdbDealing
from src.ner import NerDealing
from src.math import MathDealing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='crowdsourcing and LLM')
    parser.add_argument('--task', type=str, default="math", help='Input the task')

    args = parser.parse_args()

    task_name = args.task
    if task_name == "math":
        build_class = MathDealing()
        # build_class.human_score()
        build_class.plot()
        build_class.compare_plot()

    elif task_name == "ner":
        build_class = NerDealing()
        build_class.huamn_plot()
        build_class.gpt_plot()
        build_class.compare_plot()

    elif task_name == "imdb":
        build_class = ImdbDealing()
        build_class.plot_result()
        build_class.value_plot()
        build_class.acc_plot()
        build_class.compare_plot()

    elif task_name == "persuasiveness":
        build_class = PersuasivenessDealing()
        build_class.plot()
        build_class.compare_plot()

