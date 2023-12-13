import re
import os
import time
import openai
import json
import jsonlines
import pandas as pd
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt


class ImdbDealing():
    def __init__(self):
        self.input_file_path = "../master_project/datasets/imdb/"
        self.csv_file_path = "../master_project/datasets/imdb/imdb.csv"
        self.data_file_path = "../master_project/datasets/imdb/imdb.json"
        self.json_file_path = "../master_project/datasets/imdb/imdb_output.json"
        self.jsonl_file_path = "../master_project/datasets/imdb/imdb.jsonl"
        self.survey_file_path = "../master_project/datasets/survey_data/imdb_0811.csv"
        self.REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
        self.NO_SPACE = ""
        self.SPACE = " "

    def _preprocess_reviews(self, reviews):
        return [self.REPLACE_WITH_SPACE.sub(self.SPACE, line) for line in reviews]

    def _get_dataset(self):
        dataset_test = load_dataset("imdb", split="test", cache_dir=self.input_file_path)
        return dataset_test["text"], dataset_test["label"]

    def _deal_dataset(self):
        reviews_test, ground_truth = self._get_dataset()
        reviews_test_clean = self._preprocess_reviews(reviews_test)

        df_test_reviews_clean = pd.DataFrame(reviews_test_clean, columns=['reviews'])
        df_test_reviews_clean['target'] = ground_truth

        positive_test_reiews_imdb = df_test_reviews_clean[df_test_reviews_clean['target'] == 1]
        negative_test_reiews_imdb = df_test_reviews_clean[df_test_reviews_clean['target'] == 0]

        positive_test_reiews_imdb_gpt = positive_test_reiews_imdb.sample(n=500, random_state=525)
        negative_test_reiews_imdb_gpt = negative_test_reiews_imdb.sample(n=500, random_state=525)

        # full_reviews_imdb = pd.DataFrame({'id': []})
        full_reviews_imdb = pd.concat([positive_test_reiews_imdb_gpt, negative_test_reiews_imdb_gpt], axis=0,
                                          ignore_index=True)
        full_gpt_reviews_imdb = full_reviews_imdb.sample(frac=1).reset_index(drop=True)
        full_gpt_reviews_imdb.to_csv(self.csv_file_path, encoding='utf-8')

    def _csv_json(self):
        self._deal_dataset()
        full_dict = {}

        full_gpt_reviews_imdb = pd.read_csv(self.csv_file_path)
        full_reviews = full_gpt_reviews_imdb['reviews'].values.tolist()
        full_target = full_gpt_reviews_imdb['target'].values.tolist()

        for idx in range(len(full_reviews)):
            review = full_reviews[idx]
            full_imdb_output = {}
            target = full_target[idx]

            prefix = "Whether the following review is positive or negative: \n"

            full_imdb_output["prefix"] = prefix
            full_imdb_output["input"] = review
            full_imdb_output["answer"] = target

            full_dict[idx] = full_imdb_output

        with open(self.data_file_path, 'w') as json_file:
            json.dump(full_dict, json_file, indent=4)

    def _chatgpt(prompt):
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
                while True:
                    output = completion.choices[0].message["content"]
                    if output in ['A', 'B']:
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


    def chatgpt_output(self):
        file_read = open(self.data_file_path, 'r')
        json_file = json.load(file_read)
        shot = """
            Question: Is the following review positive or negative?
            I was trying to work out why I enjoyed this film?? Its not because of money spent on it that's for surell Did I see a painted water pistol in there? Maybe they don't have the same sort of visual effects houses in the Scotland? Or maybe they just didn't have any money? The making of clearly shows a gang of very plucky guys making a movie against the odds. Awesome! But what I really liked was the grit of the performances. Mike Michell and Patrick White play the lead parts like 2 normal guys. No Hollywood histrionics here. OK, so the effects work isn't very good. The spaceships just don't look as good as they should in todays FX world and I've seen much better free stuff on youtube. But the film holds together very well once they get to the Planet. Was this filmed in Scotland or just by a Scottish crew? Or is it just better effects work? Did they edit out the water? By the end I kinda loved this film and was disappointed when they all died.
            Choice: A. Negative
                    B. Positive
            Answer: A\n
        """
        keys_lst = json_file.keys()
        task_dict = {}


        for idx in keys_lst:
            sentence_dict = {}
            content_dict = json_file[idx]
            prefix = """
              Question: Is the following review positive or negative?
              """
            text = content_dict["input"]
            surfix = """
              Choice: A. Negative
                    B. Positive
              Answer:"""
            prompt = shot + prefix + text + surfix

            output = self._chatgpt(prompt)

            sentence_dict["idx"] = idx
            sentence_dict["output"] = output
            sentence_dict["answer"] = json_file[idx]["answer"]
            sentence_dict["text"] = text
            sentence_dict["prompt"] = prompt

            task_dict[idx] = sentence_dict

            with jsonlines.open(self.jsonl_file_path, 'a') as writer:
                writer.write(sentence_dict)

        with open(self.json_file_path, 'w') as fw:
            json.dump(task_dict, fw, indent=4)

    def _gpt_score(self):
        # 0 & A negative
        # 1 & B positive
        correct_lst = [0, 0, 0, 0, 0]
        negative_lst = [0, 0, 0, 0, 0]
        positive_lst = [0, 0, 0, 0, 0]
        total_lst = [0, 0, 0, 0, 0]
        diff_lst = [0, 0, 0, 0, 0]
        question_lst = []
        question_rate_dict, human_score, human_sum, valid_column = self._score_calculate()

        with jsonlines.open(self.jsonl_file_path, 'r') as reader:
            for line in reader:
                question_name = 'Question' + line['idx']
                if question_name != 'Question50':
                    if question_name in question_rate_dict["1"] and question_name in valid_column:
                        question_lst.append(question_name)
                        total_lst[0] += 1
                        predict = '0' if line['output'] == 'A' else '1'
                        answer = str(line['answer'])
                        if predict == '0':
                            negative_lst[0] += 1
                        else:
                            positive_lst[0] += 1

                        if answer == predict:
                            correct_lst[0] += 1
                        else:
                            diff_lst[0] += 1

                    elif question_name in question_rate_dict["2"] and question_name in valid_column:
                        question_lst.append(question_name)
                        total_lst[1] += 1
                        predict = '0' if line['output'] == 'A' else '1'
                        answer = str(line['answer'])
                        if predict == '0':
                            negative_lst[1] += 1
                        else:
                            positive_lst[1] += 1

                        if answer == predict:
                            correct_lst[1] += 1
                        else:
                            diff_lst[1] += 1

                    elif question_name in question_rate_dict["3"] and question_name in valid_column:
                        question_lst.append(question_name)
                        total_lst[2] += 1
                        predict = '0' if line['output'] == 'A' else '1'
                        answer = str(line['answer'])
                        if predict == '0':
                            negative_lst[2] += 1
                        else:
                            positive_lst[2] += 1

                        if answer == predict:
                            correct_lst[2] += 1
                        else:
                            diff_lst[2] += 1

                    elif question_name in question_rate_dict["4"] and question_name in valid_column:
                        question_lst.append(question_name)
                        total_lst[3] += 1
                        predict = '0' if line['output'] == 'A' else '1'
                        answer = str(line['answer'])
                        if predict == '0':
                            negative_lst[3] += 1
                        else:
                            positive_lst[3] += 1

                        if answer == predict:
                            correct_lst[3] += 1
                        else:
                            diff_lst[3] += 1

                    elif question_name in question_rate_dict["5"] and question_name in valid_column:
                        question_lst.append(question_name)
                        total_lst[4] += 1
                        predict = '0' if line['output'] == 'A' else '1'
                        answer = str(line['answer'])
                        if predict == '0':
                            negative_lst[4] += 1
                        else:
                            positive_lst[4] += 1

                        if answer == predict:
                            correct_lst[4] += 1
                        else:
                            diff_lst[4] += 1

        # acc = np.sum(correct_lst) / np.sum(total_lst)
        return negative_lst, positive_lst, correct_lst, diff_lst

    def _score_calculate(self):
        df = pd.read_csv(self.survey_file_path)

        column_names = df.columns
        valid_data = df.iloc[3:]
        valid_data = valid_data[valid_data['Status'] != "Survey Preview"]

        start_position = valid_data.columns.get_loc("Question0")
        end_position = valid_data.columns.get_loc("Question99")

        question_data = valid_data.iloc[:, start_position:end_position + 1]

        output_map = {"Positive": 1, "Negative": 0}

        with open(self.json_file_path, 'r') as f_read:
            answer_content = json.load(f_read)

        length = end_position - start_position + 1
        question_dict = {}
        for i in range(length):
            column_name = f"Question{i}"
            if column_name != 'Question50':
                column_content = question_data[column_name]

                for value in column_content:
                    answer = value
                    if not pd.isna(answer):
                        question_dict[column_name] = output_map[answer]

        question_avg_dict = {}
        for i in range(length):
            column_name = f"Question{i}"
            column_content = question_data[column_name]
            question_avg_dict[column_name] = []
            if column_name != 'Question50':
                for value in column_content:
                    answer = value
                    if not pd.isna(answer):
                        question_avg_dict[column_name].append(output_map[answer])

        for key in question_avg_dict.keys():
            question_avg_dict[key] = np.mean(question_avg_dict[key]) if question_avg_dict[key] != [] else 0

        valid_column = list(question_dict.keys())

        question_rate_dict = {"1": [], "2": [], "3": [], "4": [], "5": []}
        for question in list(question_avg_dict.keys()):
            correct_rate = question_avg_dict[question]
            if 0 <= correct_rate < 0.125:
                question_rate_dict["1"].append(question)
            elif 0.125 <= correct_rate < 0.375:
                question_rate_dict["2"].append(question)
            elif 0.375 <= correct_rate < 0.625:
                question_rate_dict["3"].append(question)
            elif 0.625 <= correct_rate < 0.875:
                question_rate_dict["4"].append(question)
            elif 0.875 <= correct_rate <= 1:
                question_rate_dict["5"].append(question)

        human_score = 0
        human_sum = 0

        for key in list(question_dict.keys()):
            human_sum += 1
            idx = key.replace("Question", "")
            predict = question_dict[key]
            answer = answer_content[str(idx)]['answer']

            if answer == predict:
                human_score += 1

        print("human score is", human_score / human_sum)
        return question_rate_dict, human_score, human_sum, valid_column

    def get_gptscore(self):
        question_rate_dict, human_score, human_sum, valid_column = self._score_calculate()
        correct = 0
        total = 0
        with jsonlines.open(self.jsonl_file_path, 'r') as reader:
            for line in reader:
                question_name = "Question" + line['idx']
                if question_name != 'Question50':
                    if question_name in valid_column:
                        total += 1
                        predict = '0' if line['output'] == 'A' else '1'
                        answer = str(line['answer'])

                        if answer == predict:
                            correct += 1
                        else:
                            correct = correct

        acc = correct / total

        print("gpt score is", acc)
        return correct, total

    def plot_result(self):
        question_rate_dict, human_score, human_sum, valid_column = self._score_calculate()

        correct_lst = [0, 0, 0, 0, 0]
        total_lst = [0, 0, 0, 0, 0]
        diff_lst = [0, 0, 0, 0, 0]

        with jsonlines.open(self.jsonl_file_path, 'r') as reader:
            for line in reader:
                question_name = 'Question' + line['idx']
                if question_name != 'Question50':
                    if question_name in question_rate_dict["1"] and question_name in valid_column:
                        total_lst[0] += 1
                        predict = '0' if line['output'] == 'A' else '1'
                        answer = str(line['answer'])

                        if answer == predict:
                            correct_lst[0] += 1
                        else:
                            diff_lst[0] += 1

                    elif question_name in question_rate_dict["2"] and question_name in valid_column:
                        total_lst[1] += 1
                        predict = '0' if line['output'] == 'A' else '1'
                        answer = str(line['answer'])

                        if answer == predict:
                            correct_lst[1] += 1
                        else:
                            diff_lst[1] += 1

                    elif question_name in question_rate_dict["3"] and question_name in valid_column:
                        total_lst[2] += 1
                        predict = '0' if line['output'] == 'A' else '1'
                        answer = str(line['answer'])

                        if answer == predict:
                            correct_lst[2] += 1
                        else:
                            diff_lst[2] += 1

                    elif question_name in question_rate_dict["4"] and question_name in valid_column:
                        total_lst[3] += 1
                        predict = '0' if line['output'] == 'A' else '1'
                        answer = str(line['answer'])

                        if answer == predict:
                            correct_lst[3] += 1
                        else:
                            diff_lst[3] += 1

                    elif question_name in question_rate_dict["5"] and question_name in valid_column:
                        total_lst[4] += 1
                        predict = '0' if line['output'] == 'A' else '1'
                        answer = str(line['answer'])

                        if answer == predict:
                            correct_lst[4] += 1
                        else:
                            diff_lst[4] += 1

        title_lst = ["[0, 0.125]", "[0.125, 0.375]", "[0.375, 0.625]", "[0.625, 0.875]", "[0.875, 1]"]
        for i in range(4):
            categories = ["correct", "incorrect"]
            values = [correct_lst[i], total_lst[i] - correct_lst[i]]

            plt.bar(categories, values, width=0.5)
            plt.xlabel('ChatGPT Type')
            plt.ylabel('Number')
            plt.title(f'Human Average in {title_lst[i]}')
            plt.show()

    def value_plot(self):
        negative_lst, positive_lst, correct_lst, diff_lst = self._gpt_score()
        x_labels = ["[0, 0.125]", "[0.125, 0.375]", "[0.375, 0.625]", "[0.625, 0.875]", "[0.875, 1]"]

        barWidth = 0.25

        y1 = negative_lst
        y2 = positive_lst

        r1 = np.arange(len(y1))
        r2 = [x + barWidth for x in r1]

        plt.bar(r1, y1, width=barWidth, label='Negative Number')
        plt.bar(r2, y2, width=barWidth, label='Positive Number')

        plt.legend()
        plt.xlabel('Human Average Value', fontweight='bold')
        plt.ylabel('ChatGPT Value')
        plt.title('Comparison')
        plt.xticks([r + barWidth for r in range(len(y1))], x_labels)

        plt.tight_layout()

        plt.show()

    def acc_plot(self):
        negative_lst, positive_lst, correct_lst, diff_lst = self._gpt_score()
        x_labels = ["[0, 0.125]", "[0.125, 0.375]", "[0.375, 0.625]", "[0.625, 0.875]", "[0.875, 1]"]

        barWidth = 0.25

        y1 = correct_lst
        y2 = diff_lst
        print("correct_lst", correct_lst)
        print("diff_lst", diff_lst)

        r1 = np.arange(len(y1))
        r2 = [x + barWidth for x in r1]

        plt.bar(r1, y1, width=barWidth, label='Correct Number')
        plt.bar(r2, y2, width=barWidth, label='Incorrect Number')

        plt.legend()
        plt.xlabel('Human Average Value', fontweight='bold')
        plt.ylabel('ChatGPT Value')
        plt.title('Comparison')
        plt.xticks([r + barWidth for r in range(len(y1))], x_labels)

        plt.tight_layout()

        plt.show()

    def compare_plot(self):
        gpt_correct, gpt_total = self.get_gptscore()
        question_rate_dict, human_score, human_sum, valid_column = self._score_calculate()
        x_labels = ["Correct Number", "Incorrect Number"]

        barWidth = 0.25

        y1 = [human_score, human_sum - human_score]
        y2 = [gpt_correct, (gpt_total - gpt_correct)]

        r1 = np.arange(len(y1))
        r2 = [x + barWidth for x in r1]

        plt.bar(r1, y1, width=barWidth, label='Human')
        plt.bar(r2, y2, width=barWidth, label='ChatGPT')

        plt.legend()
        plt.xlabel('Type', fontweight='bold')
        plt.ylabel('Number')
        plt.title('Crowdsourcing VS ChatGPT')
        plt.xticks([r + barWidth for r in range(len(y1))], x_labels)

        plt.tight_layout()

        plt.show()



















