import re
import time
import json
import openai
import jsonlines
import pandas as pd
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt


class MathDealing():
    def __init__(self):
        self.input_file_path = "../datasets/math/"
        self.csv_file_path = "../datasets/math/math.csv"
        self.data_file_path = "../datasets/math/math_data.json"
        self.json_file_path = "../datasets/math/math.json"
        self.jsonl_file_path = "../datasets/math/math.jsonl"
        self.survey_data_1 = "../datasets/survey_data/math_2310_renew.csv"
        self.survey_data_2 = "../datasets/survey_data/math_2310_renew.csv"
        self.answer_path = "../datasets/math/answer.txt"

        self.REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
        self.NO_SPACE = ""
        self.SPACE = " "

    def _get_dataset(self):
        return load_dataset("gsm8k", "main", split="test", cache_dir=self.input_file_path)

    def _preprocess_reviews(self, reviews):
        output = self.REPLACE_WITH_SPACE.sub(self.SPACE, reviews)

        return output
    def _deal_dataset(self):
        dataset_test_gsm8k = self._get_dataset()
        full_dict = {}

        input_text = dataset_test_gsm8k["question"][0:1000]
        answer = dataset_test_gsm8k["answer"][0:1000]

        for idx in range(len(input_text)):
            question_dict = {}
            clean_text = self.preprocess_reviews(input_text[idx])
            question_dict['text'] = clean_text
            question_dict['answer'] = answer[idx]

            full_dict[idx] = question_dict

        with open(self.data_file_path, 'w') as json_file:
            json.dump(full_dict, json_file, indent=4)

    @staticmethod
    def _is_number(x):
        try:
            int(x)
            return x
        except ValueError:
            return False

    def _chatgpt(self, prompt):
        openai.api_key = "OPENAI KEY HERE"
        while True:
            try:

                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1,
                    temperature=0
                )
                # for math
                while True:
                    num = completion.choices[0].message["content"]
                    output = self._is_number(num)
                    if output != False:
                        break

                output = completion.choices[0].message["content"]

                break
            except openai.error.APIError as e:
                print("API error:", e)
                print("Retrying in 3 seconds...")
                time.sleep(3)

            except openai.error.RateLimitError as e:
                print("RateLimit error:", e)
                print("Retrying in 3 seconds...")
                time.sleep(3)

        return output

    def gpt_output(self):
        file_read = open(self.data_file_path, 'r')
        json_file = json.load(file_read)
        shot = """
        Please directly output the answer without any explanation.
        Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
        Answer: 11

        Question: The bakers at the Beverly Hills Bakery baked 200 loaves of bread on Monday morning. They sold 93 loaves in the morning and 39 loaves in the afternoon.A grocery store returned 6 unsold loaves. How many loaves of bread did they have left?
        Answer: 74\n
        """
        keys_lst = list(json_file.keys())
        task_dict = {}

        for idx in keys_lst:
            sentence_dict = {}
            content_dict = json_file[idx]
            text = "Question: " + content_dict["text"] + "\nAnswer: "
            prompt = shot + text

            output = self._chatgpt(prompt)

            sentence_dict["idx"] = idx
            sentence_dict["output"] = output
            sentence_dict["answer"] = content_dict["answer"]
            sentence_dict["text"] = text

            task_dict[idx] = sentence_dict

            with jsonlines.open(self.jsonl_file_path,'a') as writer:
                writer.write(sentence_dict)

        with open(self.json_file_path, 'w') as fw:
            json.dump(task_dict, fw, indent=4)

    def _deal_survey(self):
        df = pd.read_csv(self.survey_data_1)
        column_names = df.columns
        valid_data = df.iloc[2:17]
        filtered_data = valid_data[valid_data['Status'] != "Survey Preview"]

        df_1 = pd.read_csv(self.survey_data_2)
        column_names_1 = df_1.columns
        valid_data_1 = df_1.iloc[18:73]
        filtered_data_1 = valid_data_1[valid_data_1['Status'] != "Survey Preview"]

        valid_data_combine = pd.concat([filtered_data, filtered_data_1], ignore_index=True)

        start_position = valid_data_combine.columns.get_loc("Question0")
        end_position = valid_data_combine.columns.get_loc("Question99")
        question_data = valid_data_combine.iloc[:, start_position:end_position + 1]

        question_dict = {'Question50': []}
        length = end_position - start_position + 1

        question_lst = []
        for i in range(length):
            if i != 50:
                column_name = f"Question{i}"
                column_content = question_data[column_name]
                num = 0
                score = 0
                answer_lst = []

                for value in column_content:
                    answer = value
                    if not pd.isna(value):
                        answer_lst.append(answer)
                question_dict[column_name] = answer_lst
        return question_dict

    def _gpt_score(self):
        correct = 0
        incorrect = 0
        total = 0

        with jsonlines.open(self.jsonl_file_path , 'r') as reader:
            for line in reader.iter():
                name = f"Question{line['idx']}"
                output = line['output']

                answer = line['answer'].split("#### ")[1]
                if output != "$":
                    total += 1
                    # print("output", output)
                    if output == answer:
                        correct += 1
                    else:
                        incorrect += 1

        print("acc", correct / total)

    def human_score(self):
        question_dict = self._deal_survey()
        answer_file = open(self.answer_path, "r")
        answer = answer_file.readlines()
        answer_lst = [i.split("\n#### ")[1] for i in answer]
        correct_rate_dict = {"1": [], "2": [], "3": [], "4": [], "5": []}
        correct_dict = {}
        total_number = 0

        for (key, answer) in zip(list(question_dict.keys()), answer_lst):
            correct_dict[key] = {}
            correct_dict[key]['total_number'] = len(question_dict[key])
            total_number += len(question_dict[key])
            nominator = len([i for i in question_dict[key] if i == answer])
            avg = nominator / len(question_dict[key]) if len(question_dict[key]) != 0 else 0
            correct_dict[key]['correct_rate'] = avg
            correct_dict[key]['correct_number'] = nominator
            correct_dict[key]['false_number'] = len(question_dict[key]) - nominator

        for question in list(correct_dict.keys()):
            correct_rate = correct_dict[question]['correct_rate']
            if correct_rate in [0, 0.126]:
                correct_rate_dict["1"].append(question)
            elif correct_rate in [0.126, 0.376]:
                correct_rate_dict["2"].append(question)
            elif correct_rate in [0.376, 0.626]:
                correct_rate_dict["3"].append(question)
            elif correct_rate in [0.626, 0.876]:
                correct_rate_dict["4"].append(question)
            elif correct_rate in [0.876, 1]:
                correct_rate_dict["5"].append(question)
        return correct_rate_dict

    def plot(self):
        correct_rate_dict = self.human_score()
        title_lst = ["[0, 0.126]", "[0.126, 0.376]", "[0.376, 0.626]", "[0.626, 0.876]", "[0.876, 1]"]
        for i in range(1, 5):
            correct = 0
            incorrect = 0
            total = 0
            question_lst = correct_rate_dict[str(i)]

            with jsonlines.open(self.jsonl_file_path, 'r') as reader:
                for line in reader.iter():
                    name = f"Question{line['idx']}"
                    if name in question_lst:
                        output = line['output']

                        answer = line['answer'].split("#### ")[1]
                        if output != "$":
                            total += 1
                            if output == answer:
                                correct += 1
                            else:
                                incorrect += 1

            categories = ["Correct Number", "Incorrect Number"]
            values = [correct, incorrect]

            plt.bar(categories, values, width=0.5)
            plt.xlabel('Values')
            plt.ylabel('Number')
            plt.title(f'GPT(when human in {title_lst[i-1]}')
            plt.show()




