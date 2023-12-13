import json
import time
import openai
import jsonlines
import pandas as pd
import numpy as np

from collections import Counter
import matplotlib.pyplot as plt


class PersuasivenessDealing():
    def __init__(self):
        self.file_dict = "../master_project/datasets/persuasiveness"
        self.input_file_path = "../master_project/datasets/persuasiveness/persuasiveness.json"
        self.csv_file_path = "../master_project/datasets/persuasiveness/persuasiveness.csv"
        self.json_file_path = "../master_project/datasets/persuasiveness/persuasiveness_output.json"
        self.jsonl_file_path = "../master_project/datasets/persuasiveness/persuasiveness.jsonl"
        self.survey_data = "../master_project/datasets/survey_data/persuasiveness_0811.csv"
        self.map_dict = {"Not Persuasive": 1, "Somewhat Persuasive": 2, "Persuasive": 3, "Very Persuasive": 4}
        self.choice_map_dict = {"A": 1, "B": 2, "C": 3, "D": 4}

    @staticmethod
    def _chatgpt(prompt):
        openai.api_key = "YOUR OPENAI KEY HERE"
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
                    if output in ['A', 'B', 'C', 'D']:
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
        file_read = open(self.input_file_path, 'r')
        json_file = json.load(file_read)
        shot = """
        Question: How persuasive is this essay?
        Essay: Gun is definitely not a kind of normal good in our society, which is strictly controlled in most countries. However, gun control in some countries is relatively loose than in others. Therefore, this is still a controversial topic around the world. From my point of view, it is sensible to enforce the gun control. The first factor considered is the risk of accidents with guns. The accidents cannot be completely avoided although only those who have gun licenses can purchase guns. For example, we hear some news from time to time that someone was hurt by accident during the hunting. So the fewer guns possessed, the fewer accidents would happened, our surroundings would be safer with lower accident rates. Secondly, most violent crimes are related to the abuse of guns, especially in some countries where guns are available for people. Eventually, guns will create a violent society if the trend continues. Take an example, in American, young adults and even juveniles can get access to guns, which leads to the tragedies of school gun shooting. What is worse, some terrorists are able to possess more advanced weapons than the police, which makes citizens always live in danger. Thirdly, the possession of guns can also raise the rates of suicide. In the US, firearms remain the most common method of suicide, accounting for about 50 per cent. Unfortunately, there is an increasing trend of adolescent suicides and suicides among those age 75 and over. In conclusion, considering the rise in accident rates, violent crime rates and suicide rates, I support that the guns should be strictly limited and the government should enforce a series of laws to prevent our societies from violence.
        Choice: A. Not Persuasive
                B. Somewhat Persuasive
                C. Persuasive
                D. Very Persuasive
        Answer: B\n"""
        keys_lst = json_file.keys()
        task_dict = {}

        for idx in keys_lst:
            sentence_dict = {}
            content_dict = json_file[idx]
            prefix = """Question: How persuasive is this essay?\n"""
            text = """Essay: """ + content_dict["full_prompt"]
            surfix = """
        Choice: A. Not Persuasive
              B. Somewhat Persuasive
              C. Persuasive
              D. Very Persuasive
        Answer:"""
            prompt = shot + prefix + text + surfix

            output = self._chatgpt(prompt)

            sentence_dict["idx"] = idx
            sentence_dict["output"] = output
            sentence_dict["text"] = text
            sentence_dict["prefix"] = prompt

            task_dict[idx] = sentence_dict

            with jsonlines.open(self.jsonl_file_path, 'a') as writer:
                writer.write(sentence_dict)

        with open(self.json_file_path, 'w') as fw:
            json.dump(task_dict, fw, indent=4)

    def _deal_survey(self):
        forced_quetion = "Question57"

        df = pd.read_csv(self.survey_data)
        column_names = df.columns
        valid_data = df.iloc[4:104]
        filtered_data = valid_data[valid_data['Status'] != "Survey Preview"]

        start_position = filtered_data.columns.get_loc("Question22")
        end_position = filtered_data.columns.get_loc("Question2")
        question_data = filtered_data.iloc[:, start_position:end_position + 1]

        question_dict = {"Question36": [], "Question90": []}
        invalid_question_name = []
        for i in range(1, 90):
            if i not in [36, 90]:
                sum_lst = []
                column_name_display = f"Question0{i}" if i in range(1, 10) else f"Question{i}"
                column_name = f"Question{i}"
                column_content = question_data[column_name]
                num = 0
                for value in column_content:
                    answer = value
                    if not pd.isna(value):
                        num += 1
                        score = self.map_dict[answer]
                        sum_lst.append(score)
                if not sum_lst:
                    invalid_question_name.append(column_name)
                else:
                    question_dict[column_name_display] = sum_lst

        gather_keys = {}
        for key, value in question_dict.items():
            v = np.mean(value) if value != [] else 0
            if key not in invalid_question_name:
                if v in gather_keys:
                    gather_keys[v].append(key) if key not in gather_keys[v] else gather_keys[v]
                else:
                    gather_keys[v] = [key]
        return gather_keys, question_dict, invalid_question_name

    def cal_score(self):
        gather_keys, question_dict, invalid_question_name = self._deal_survey()
        with open(self.json_file_path, 'r') as f_read:
            json_data = json.load(f_read)

        essay_name_lst = list(json_data.keys())
        gpt_score_dict = {}
        gather_gpt_dict = {}
        refurnised_dict = {1: [], 2: [], 3: [], 4: []}

        for essay in essay_name_lst:
            essay_content = json_data[essay]
            essay_name = essay_content['idx'].replace("essay", "Question")
            output = essay_content['output']
            score_gpt = self.choice_map_dict[output]
            gpt_score_dict[essay_name] = score_gpt

        for key in list(gather_keys.keys()):
            question_lst = gather_keys[key]
            gather_gpt_dict[key] = [gpt_score_dict[question] for question in question_lst]

        for key in gather_gpt_dict.keys():
            if 1 <= key < 1.5:
                refurnised_dict[1].extend(gather_gpt_dict[key])
            elif 1.5 <= key < 2.5:
                refurnised_dict[2].extend(gather_gpt_dict[key])
            elif 2.5 <= key < 3.5:
                refurnised_dict[3].extend(gather_gpt_dict[key])
            elif 3.5 <= key <= 4:
                refurnised_dict[4].extend(gather_gpt_dict[key])

        return gpt_score_dict, question_dict, refurnised_dict

    def plot(self):
        gpt_score_dict, question_dict, refurnised_dict = self.cal_score()
        gpt_value = []

        for i in range(1, 91):
            key_name = f"Question0{i}" if i in range(1, 10) else f"Question{i}"
            gpt_value.append(float(gpt_score_dict[key_name]))

        question_values = list(question_dict.values())
        human_lst = []
        for i in question_values:
            if i != []:
                human_lst.extend(i)

        human_dict = dict(Counter(human_lst))
        gpt_dict = dict(Counter(gpt_value))

        categories = [1, 2, 3, 4]
        values = list(human_dict.values())

        plt.bar(categories, values)
        plt.xticks(categories, [str(int(cat)) for cat in categories])
        plt.xlabel('Values')
        plt.ylabel('Number')
        plt.title('Human')
        plt.show()

        categories = list(gpt_dict.keys())
        values = list(gpt_dict.values())

        plt.bar(categories, values)
        plt.xticks(categories, [str(int(cat)) for cat in categories])
        plt.xlabel('Values')
        plt.ylabel('Number')
        plt.title('GPT')
        plt.show()

        categories = ["Less Persuasiveness", "Persuasiveness", "Somewhat Persuasiveness", "Vary Persuasiveness"]
        feature1 = [10, 14, 47, 4]
        feature2 = [2, 0, 83, 5]
        bar_width = 0.35
        x = np.arange(len(categories))

        plt.bar(x - bar_width / 2, feature1, width=bar_width, label='Crowdsourcing')
        plt.bar(x + bar_width / 2, feature2, width=bar_width, label='ChatGPT')

        plt.xticks(x, categories, rotation=15)

        plt.xlabel('Persuasiveness Type')
        plt.ylabel('Number')
        plt.title('Crowdsourcing VS ChatGPT')
        plt.legend()
        plt.show()

    def compare_plot(self):
        gpt_score_dict, question_dict, refurnised_dict = self.cal_score()
        keys = list(refurnised_dict.keys())
        key_name = ''

        for key in sorted(keys):
            gpt_lst = refurnised_dict[key]
            value_counts = dict(Counter(gpt_lst))
            x_label = [0, 1, 2, 3, 4]
            # print(value_counts)
            y_label = [0, 0, 0, 0, 0]
            for got_key in value_counts.keys():
                y_label[int(got_key)] = value_counts[got_key]
            # print(y_label)
            if key == 1:
                key_name = '[1, 1.5]'
            elif key == 2:
                key_name = '[1.5, 2.5]'
            elif key == 3:
                key_name = '[2.5, 3.5]'
            elif key == 4:
                key_name = '[3.5, 4.1]'

            plt.bar(x_label, y_label)
            plt.xlabel('GPT Values')
            plt.ylabel('Number')
            plt.title(f'Human Average in {key_name}')
            plt.show()
