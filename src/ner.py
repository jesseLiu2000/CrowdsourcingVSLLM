import os
import json
import time
import openai
import random
import jsonlines
import pandas as pd
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt


class NerDealing():
    def __init__(self):
        self.file_dict = "../master_project/datasets/ner"
        self.input_file_path = "../master_project/datasets/ner/ner.json"
        self.csv_file_path = "../master_project/datasets/ner/ner.csv"
        self.json_file_path = "../master_project/datasets/ner/ner_output.json"
        self.jsonl_file_path = "../master_project/datasets/ner/ner.jsonl"
        self.survey_data = "../master_project/datasets/survey_data/ner_0811.csv"

    def _get_dataset(self):
        return load_dataset("conll2003", split="train", cache_dir=self.file_dict)

    def _bio_to_list(self, bio_format):
        entity_list = []
        current_entity = None

        for token, tag in bio_format:
            if tag.startswith('B-'):
                if current_entity:
                    entity_list.append(current_entity)
                current_entity = {'entity': token, 'tag': tag[2:]}
            elif tag.startswith('I-'):
                if current_entity and current_entity['tag'] == tag[2:]:
                    current_entity['entity'] += ' ' + token
            else:
                if current_entity:
                    entity_list.append(current_entity)
                current_entity = None

        if current_entity:
            entity_list.append(current_entity)

        return entity_list

    def _deal_dataset(self):
        dataset = self._get_dataset()
        random_numbers = random.sample(range(1, 14042), 1000)
        mapping_dict = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7,
                        'I-MISC': 8}

        token_lst = dataset['tokens']
        ner_lst = dataset['ner_tags']

        total_lst = []
        full_dict = {}

        for idx in random_numbers:

            ners = ner_lst[idx]
            tokens = token_lst[idx]

            bio_format = []
            sentence_dict = {}
            ner_tag_dict = {}

            loc_lst = []
            per_lst = []
            org_lst = []
            misc_lst = []

            text = " ".join(tokens)
            sentence_dict["idx"] = idx
            sentence_dict["text"] = text

            for n, t in zip(ners, tokens):
                tag = list(mapping_dict.keys())[list(mapping_dict.values()).index(n)]
                ner_triple = (t, tag)
                bio_format.append(ner_triple)

            results = self._bio_to_list(bio_format)
            total_lst.append(results)

            for entity_dict in results:
                entity_tag = entity_dict['tag']
                if entity_tag == "ORG":
                    org_lst.append(entity_dict['entity'])
                elif entity_tag == "LOC":
                    loc_lst.append(entity_dict['entity'])
                elif entity_tag == "PER":
                    per_lst.append(entity_dict['entity'])
                elif entity_tag == "MISC":
                    misc_lst.append(entity_dict['entity'])

            ner_tag_dict["ORG"] = org_lst
            ner_tag_dict["MISC"] = misc_lst
            ner_tag_dict["PER"] = per_lst
            ner_tag_dict["LOC"] = loc_lst

            sentence_dict["answer"] = ner_tag_dict

            full_dict[idx] = sentence_dict

        with open(self.json_file_path, 'w') as json_file:
            json.dump(full_dict, json_file, indent=4)

    @staticmethod
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
        file_read = open(self.json_file_path, 'r')
        json_file = json.load(file_read)
        shot = """
        In the sentence below, extract required entities should be filled in the json format.
          - organization named entity
          - location named entity
          - person named entity
          - miscellaneous named entity.
        Question: During his visit to Slovenia , Kwasniewski is also scheduled to meet Prime Minister Janez Drnovsek , representatives of Slovenian political parties and representatives of the Chamber of Economy .
        Answer: {"organization named entity": ["Chamber of Economy"], "person named entity":["Kwasniewski","Janez Drnovsek"], "location named entity":["Slovenia"], "miscellaneous named entity":["Slovenian"]}\n"""
        keys_lst = list(json_file.keys())
        task_dict = {}

        for idx in keys_lst:
            sentence_dict = {}
            content_dict = json_file[idx]
            prefix = """
            In the sentence below, extract required entities should be filled in the json format.
            - organization named entity
            - location named entity
            - person named entity
            - miscellaneous named entity.
            """
            text = """Question: """ + content_dict["text"] + """\nAnswer: """
            prompt = shot + prefix + text

            output = self._chatgpt(prompt)

            sentence_dict["idx"] = idx
            sentence_dict["output"] = output
            sentence_dict["answer"] = json_file[idx]["answer"]
            sentence_dict["text"] = text
            sentence_dict["prefix"] = prompt

            task_dict[idx] = sentence_dict

            with jsonlines.open(self.jsonl_file_path, 'a') as writer:
                writer.write(sentence_dict)

        with open(self.json_file_path, 'w') as fw:
            json.dump(task_dict, fw, indent=4)

    @staticmethod
    def _intersection_length(lst1, lst2):
        lst1_modify = []
        for value in lst1:
            value = value.replace('"', "")
            lst1_modify.append(value)

        lst3 = [value for value in lst1_modify if value in lst2]
        return len(lst3)

    def survey_deal(self):
        df = pd.read_csv(self.survey_data)
        column_names = df.columns.tolist()

        valid_data = df.iloc[2:72]
        valid_data = valid_data[valid_data['Status'] != "Survey Preview"]

        start_position = valid_data.columns.get_loc("Question1304_1")
        end_position = valid_data.columns.get_loc("Question12782_4")
        column_names = column_names[start_position: end_position + 1]
        question_data = valid_data.iloc[:, start_position:end_position + 1]

        with open(self.json_file_path, 'r') as f_read:
            answer_content = json.load(f_read)

        ner_dict = {'1': "ORG", '2': "PER", '3': "LOC", '4': "MISC"}

        prefixes = set([q.rsplit('_', 1)[0] for q in column_names if q != 'Question5102' and q != 'Question4808'])
        combined_questions = [prefix for prefix in prefixes]

        length = start_position - end_position + 1
        question_dict = {}
        sentence_dict = {}

        for i in combined_questions:
            column_name = i
            question_index = column_name.replace("Question", "")
            answer = answer_content[question_index]['answer']
            sentence_dict[question_index] = {}
            sentence_dict[question_index]['output'] = {}
            sentence_dict[question_index]['answer'] = answer

            for j in range(1, 5):
                predict_type = str(j)
                question_name = column_name + "_" + str(j)
                column_content = question_data[question_name]

                if predict_type == '1':
                    for value in column_content:
                        if not pd.isna(value):
                            answer = value
                            type_name = ner_dict[predict_type]
                            sentence_dict[question_index]['output'][type_name] = answer.split(",")

                elif predict_type == '2':
                    for value in column_content:
                        answer = value
                        if not pd.isna(value):
                            type_name = ner_dict[predict_type]
                            sentence_dict[question_index]['output'][type_name] = answer.split(",")

                elif predict_type == '3':
                    for value in column_content:
                        answer = value
                        if not pd.isna(value):
                            type_name = ner_dict[predict_type]
                            sentence_dict[question_index]['output'][type_name] = answer.split(",")

                elif predict_type == '4':
                    for value in column_content:
                        answer = value
                        if not pd.isna(value):
                            type_name = ner_dict[predict_type]
                            sentence_dict[question_index]['output'][type_name] = answer.split(",")

        return sentence_dict

    def huamn_plot(self):
        sentence_dict = self.survey_deal()
        rec_lst = []
        pre_lst = []
        total_correct_answer_lst = []

        org_rec_lst = []
        per_rec_lst = []
        loc_rec_lst = []
        misc_rec_lst = []

        org_pre_lst = []
        per_pre_lst = []
        loc_pre_lst = []
        misc_pre_lst = []

        f1_lst = []
        org_f1_lst = []
        per_f1_lst = []
        loc_f1_lst = []
        misc_f1_lst = []

        org_correct_lst = []

        org_correct_answer_lst = []
        per_correct_answer_lst = []
        loc_correct_answer_lst = []
        misc_correct_answer_lst = []

        org_answer_length_lst = []
        per_answer_length_lst = []
        loc_answer_length_lst = []
        misc_answer_length_lst = []

        org_predict_length_lst = []
        per_predict_length_lst = []
        loc_predict_length_lst = []
        misc_predict_length_lst = []

        keys_lst = list(sentence_dict.keys())
        # for line, answer_line in zip(f.iter(), a.iter()):
        for key in keys_lst:
            line = sentence_dict[key]

            correct_answer = 0
            org_correct = 0
            per_correct = 0
            loc_correct = 0
            misc_correct = 0

            answer = line["answer"]
            predcit = line["output"]

            output_keys = list(predcit.keys())

            if predcit != {}:

                org_predict = len(predcit["ORG"]) if "ORG" in output_keys else 0
                loc_predict = len(predcit["LOC"]) if "LOC" in output_keys else 0
                per_predict = len(predcit["PER"]) if "PER" in output_keys else 0
                misc_predict = len(predcit["MISC"]) if "MISC" in output_keys else 0

                org_answer = len(answer["ORG"])
                loc_answer = len(answer["LOC"])
                per_answer = len(answer["PER"])
                misc_answer = len(answer["MISC"])

                org_correct = self._intersection_length(predcit["ORG"], answer["ORG"]) if "ORG" in output_keys else 0
                per_correct = self._intersection_length(predcit["PER"], answer["PER"]) if "PER" in output_keys else 0
                loc_correct = self._intersection_length(predcit["LOC"], answer["LOC"]) if "LOC" in output_keys else 0
                misc_correct = self._intersection_length(predcit["MISC"], answer["MISC"]) if "MISC" in output_keys else 0

                predict_length = org_predict + loc_predict + per_predict + misc_predict
                answer_length = org_answer + loc_answer + per_answer + misc_answer
                correct_answer = org_correct + per_correct + loc_correct + misc_correct

                # print("predict_length", loc_predict)
                # print("answer_length", loc_answer)
                # print("correct_answer", correct_answer)

            else:
                org_predict = 0
                loc_predict = 0
                per_predict = 0
                misc_predict = 0

                org_answer = len(answer["ORG"])
                loc_answer = len(answer["LOC"])
                per_answer = len(answer["PER"])
                misc_answer = len(answer["MISC"])

                org_correct = 0
                per_correct = 0
                loc_correct = 0
                misc_correct = 0

                predict_length = 0
                answer_length = len(answer["ORG"]) + len(answer["LOC"]) + len(answer["PER"]) + len(answer["MISC"])
                correct_answer = 0

            rec = correct_answer / answer_length if answer_length > 0 else 0
            pre = correct_answer / predict_length if predict_length > 0 else 0
            f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0

            total_correct_answer_lst.append(correct_answer)

            org_rec = org_correct / org_answer if org_answer > 0 else 0
            per_rec = per_correct / per_answer if per_answer > 0 else 0
            loc_rec = loc_correct / loc_answer if loc_answer > 0 else 0
            misc_rec = misc_correct / misc_answer if misc_answer > 0 else 0

            # org_correct_lst.append(org_correct)

            org_rec_lst.append(org_rec)
            per_rec_lst.append(per_rec)
            loc_rec_lst.append(loc_rec)
            misc_rec_lst.append(misc_rec)

            org_pre = org_correct / org_predict if org_predict > 0 else 0
            per_pre = per_correct / per_predict if per_predict > 0 else 0
            loc_pre = loc_correct / loc_predict if loc_predict > 0 else 0
            misc_pre = misc_correct / misc_predict if misc_predict > 0 else 0

            org_pre_lst.append(org_rec)
            per_pre_lst.append(per_pre)
            loc_pre_lst.append(loc_pre)
            misc_pre_lst.append(misc_pre)

            org_f1 = 2 * org_pre * org_rec / (org_pre + org_rec) if (org_pre + org_rec) > 0 else 0
            per_f1 = 2 * per_pre * per_rec / (per_pre + per_rec) if (per_pre + per_rec) > 0 else 0
            loc_f1 = 2 * loc_pre * loc_rec / (loc_pre + loc_rec) if (loc_pre + loc_rec) > 0 else 0
            misc_f1 = 2 * misc_pre * misc_rec / (misc_pre + misc_rec) if (misc_pre + misc_rec) > 0 else 0

            rec_lst.append(rec)
            pre_lst.append(pre)
            if answer_length != 0:
                f1_lst.append(f1)

            if org_answer != 0:
                org_f1_lst.append(org_f1)

            if per_answer != 0:
                per_f1_lst.append(per_f1)

            if loc_answer != 0:
                loc_f1_lst.append(loc_f1)

            if misc_answer != 0:
                misc_f1_lst.append(misc_f1)

                org_correct_answer_lst.append(org_correct)
                per_correct_answer_lst.append(per_correct)
                loc_correct_answer_lst.append(loc_correct)
                misc_correct_answer_lst.append(misc_correct)

                org_answer_length_lst.append(org_answer)
                per_answer_length_lst.append(per_answer)
                loc_answer_length_lst.append(loc_answer)
                misc_answer_length_lst.append(misc_answer)

                org_predict_length_lst.append(org_predict)
                per_predict_length_lst.append(per_predict)
                loc_predict_length_lst.append(loc_predict)
                misc_predict_length_lst.append(misc_predict)

            # break
        print("=" * 100)
        print("ALL F1 is ", np.mean(f1_lst))
        # print("total rec", np.mean(rec_lst))
        # print("total pre", np.mean(pre_lst))

        print("=" * 100)
        print("ORG F1 is ", np.mean(org_f1_lst))
        # print("org_rec", np.mean(org_rec_lst))
        # print("org_pre", np.mean(org_pre_lst))

        print("=" * 100)
        print("PER F1 is ", np.mean(per_f1_lst))
        # print("per_rec", np.mean(per_rec_lst))
        # print("per_pre", np.mean(per_pre_lst))

        print("=" * 100)
        print("LOC F1 is ", np.mean(loc_f1_lst))
        # print("loc_rec", np.mean(loc_rec_lst))
        # print("loc_pre", np.mean(loc_pre_lst))

        print("=" * 100)
        print("MISC F1 is ", np.mean(misc_f1_lst))
        # print("misc_rec", np.mean(misc_rec_lst))
        # print("misc_pre", np.mean(misc_pre_lst))

        # print("org_correct_lst", org_correct_lst)
        # print("total_correct  ", total_correct)

        print("=" * 100)

        print("org_correct_answer_lst", np.sum(org_correct_answer_lst))
        print("org_answer_length_lst", np.sum(org_answer_length_lst))
        print("org_predict_length_lst", np.sum(org_predict_length_lst))
        print("TP is", np.sum(org_correct_answer_lst))
        print("FN is", np.sum(org_answer_length_lst) - np.sum(org_correct_answer_lst))
        print("FP is", np.sum(org_predict_length_lst) - np.sum(org_correct_answer_lst))

        print("=" * 100)
        print("per_correct_answer_lst", np.sum(per_correct_answer_lst))
        print("per_answer_length_lst", np.sum(per_answer_length_lst))
        print("per_predict_length_lst", np.sum(per_predict_length_lst))
        print("TP is", np.sum(per_correct_answer_lst))
        print("FN is", np.sum(per_answer_length_lst) - np.sum(per_correct_answer_lst))
        print("FP is", np.sum(per_predict_length_lst) - np.sum(per_correct_answer_lst))

        print("=" * 100)
        print("loc_correct_answer_lst", np.sum(loc_correct_answer_lst))
        print("loc_answer_length_lst", np.sum(loc_answer_length_lst))
        print("loc_predict_length_lst", np.sum(loc_predict_length_lst))
        print("TP is", np.sum(loc_correct_answer_lst))
        print("FN is", np.sum(loc_answer_length_lst) - np.sum(loc_correct_answer_lst))
        print("FP is", np.sum(loc_predict_length_lst) - np.sum(loc_correct_answer_lst))

        print("=" * 100)
        print("misc_correct_answer_lst", np.sum(misc_correct_answer_lst))
        print("misc_answer_length_lst", np.sum(misc_answer_length_lst))
        print("misc_predict_length_lst", np.sum(misc_predict_length_lst))
        print("TP is", np.sum(misc_correct_answer_lst))
        print("FN is", np.sum(misc_answer_length_lst) - np.sum(misc_correct_answer_lst))
        print("FP is", np.sum(misc_predict_length_lst) - np.sum(misc_correct_answer_lst))

        score_lst = [np.mean(f1_lst), np.mean(org_f1_lst), np.mean(per_f1_lst), np.mean(loc_f1_lst),
                     np.mean(misc_f1_lst)]

        categories = ["ALL", "ORG", "PER", "LOC", "MISC"]
        values = score_lst

        plt.bar(categories, values)
        plt.xlabel('Values')
        plt.ylabel('F1 Score')
        plt.title('NER')
        plt.show()

        return score_lst

    def gpt_plot(self):
        sentence_dict = self.survey_deal()
        sentence_keys = list(sentence_dict.keys())
        rec_lst = []
        pre_lst = []
        total_correct_answer_lst = []

        org_rec_lst = []
        per_rec_lst = []
        loc_rec_lst = []
        misc_rec_lst = []

        org_pre_lst = []
        per_pre_lst = []
        loc_pre_lst = []
        misc_pre_lst = []

        f1_lst = []
        org_f1_lst = []
        per_f1_lst = []
        loc_f1_lst = []
        misc_f1_lst = []

        org_correct_lst = []

        org_correct_answer_lst = []
        per_correct_answer_lst = []
        loc_correct_answer_lst = []
        misc_correct_answer_lst = []

        org_answer_length_lst = []
        per_answer_length_lst = []
        loc_answer_length_lst = []
        misc_answer_length_lst = []

        org_predict_length_lst = []
        per_predict_length_lst = []
        loc_predict_length_lst = []
        misc_predict_length_lst = []

        file_path = self.jsonl_file_path
        f = jsonlines.open(file_path)

        for line in f.iter():
            idx = line['idx']
            if idx in sentence_keys:
                correct_answer = 0
                org_correct = 0
                per_correct = 0
                loc_correct = 0
                misc_correct = 0

                answer = line["answer"]
                predict = line["output"]

                # print(answer)
                # print(predcit)

                # assert answer_line["text"] == line["text"]

                # print(predcit)
                if predict != {}:
                    print(predict)
                    # print(answer)
                    org_predict = len(predict["ORG"]) if predict["ORG"] != [''] else 0
                    loc_predict = len(predict["LOC"]) if predict["LOC"] != [''] else 0
                    per_predict = len(predict["PER"]) if predict["PER"] != [''] else 0
                    misc_predict = len(predict["MISC"]) if predict["MISC"] != [''] else 0

                    org_answer = len(answer["ORG"])
                    loc_answer = len(answer["LOC"])
                    per_answer = len(answer["PER"])
                    misc_answer = len(answer["MISC"])

                    org_correct = self._intersection_length(predict["ORG"], answer["ORG"])
                    per_correct = self._intersection_length(predict["PER"], answer["PER"])
                    loc_correct = self._intersection_length(predict["LOC"], answer["LOC"])
                    misc_correct = self._intersection_length(predict["MISC"], answer["MISC"])

                    predict_length = org_predict + loc_predict + per_predict + misc_predict
                    answer_length = org_answer + loc_answer + per_answer + misc_answer
                    correct_answer = org_correct + per_correct + loc_correct + misc_correct

                    # print("predict_length", loc_predict)
                    # print("answer_length", loc_answer)
                    # print("correct_answer", correct_answer)

                else:
                    org_predict = 0
                    loc_predict = 0
                    per_predict = 0
                    misc_predict = 0

                    org_answer = len(answer["ORG"])
                    loc_answer = len(answer["LOC"])
                    per_answer = len(answer["PER"])
                    misc_answer = len(answer["MISC"])

                    org_correct = 0
                    per_correct = 0
                    loc_correct = 0
                    misc_correct = 0

                    predict_length = 0
                    answer_length = len(answer["ORG"]) + len(answer["LOC"]) + len(answer["PER"]) + len(answer["MISC"])
                    correct_answer = 0

            rec = correct_answer / answer_length if answer_length > 0 else 0
            pre = correct_answer / predict_length if predict_length > 0 else 0
            f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0

            total_correct_answer_lst.append(correct_answer)

            org_rec = org_correct / org_answer if org_answer > 0 else 0
            per_rec = per_correct / per_answer if per_answer > 0 else 0
            loc_rec = loc_correct / loc_answer if loc_answer > 0 else 0
            misc_rec = misc_correct / misc_answer if misc_answer > 0 else 0

            # org_correct_lst.append(org_correct)

            org_rec_lst.append(org_rec)
            per_rec_lst.append(per_rec)
            loc_rec_lst.append(loc_rec)
            misc_rec_lst.append(misc_rec)

            org_pre = org_correct / org_predict if org_predict > 0 else 0
            per_pre = per_correct / per_predict if per_predict > 0 else 0
            loc_pre = loc_correct / loc_predict if loc_predict > 0 else 0
            misc_pre = misc_correct / misc_predict if misc_predict > 0 else 0

            org_pre_lst.append(org_rec)
            per_pre_lst.append(per_pre)
            loc_pre_lst.append(loc_pre)
            misc_pre_lst.append(misc_pre)

            org_f1 = 2 * org_pre * org_rec / (org_pre + org_rec) if (org_pre + org_rec) > 0 else 0
            per_f1 = 2 * per_pre * per_rec / (per_pre + per_rec) if (per_pre + per_rec) > 0 else 0
            loc_f1 = 2 * loc_pre * loc_rec / (loc_pre + loc_rec) if (loc_pre + loc_rec) > 0 else 0
            misc_f1 = 2 * misc_pre * misc_rec / (misc_pre + misc_rec) if (misc_pre + misc_rec) > 0 else 0

            rec_lst.append(rec)
            pre_lst.append(pre)
            if answer_length != 0:
                f1_lst.append(f1)

                # break

            if org_answer != 0:
                org_f1_lst.append(org_f1)

            if per_answer != 0:
                per_f1_lst.append(per_f1)

            if loc_answer != 0:
                loc_f1_lst.append(loc_f1)

            if misc_answer != 0:
                misc_f1_lst.append(misc_f1)

                org_correct_answer_lst.append(org_correct)
                per_correct_answer_lst.append(per_correct)
                loc_correct_answer_lst.append(loc_correct)
                misc_correct_answer_lst.append(misc_correct)

                org_answer_length_lst.append(org_answer)
                per_answer_length_lst.append(per_answer)
                loc_answer_length_lst.append(loc_answer)
                misc_answer_length_lst.append(misc_answer)

                org_predict_length_lst.append(org_predict)
                per_predict_length_lst.append(per_predict)
                loc_predict_length_lst.append(loc_predict)
                misc_predict_length_lst.append(misc_predict)

        gpt_score_lst = [np.mean(f1_lst), np.mean(org_f1_lst), np.mean(per_f1_lst), np.mean(loc_f1_lst),
                         np.mean(misc_f1_lst)]
        # break
        print("=" * 100)
        print("ALL F1 is ", np.mean(f1_lst))
        # print("total rec", np.mean(rec_lst))
        # print("total pre", np.mean(pre_lst))

        print("=" * 100)
        print("ORG F1 is ", np.mean(org_f1_lst))
        # print("org_rec", np.mean(org_rec_lst))
        # print("org_pre", np.mean(org_pre_lst))

        print("=" * 100)
        print("PER F1 is ", np.mean(per_f1_lst))
        # print("per_rec", np.mean(per_rec_lst))
        # print("per_pre", np.mean(per_pre_lst))

        print("=" * 100)
        print("LOC F1 is ", np.mean(loc_f1_lst))
        # print("loc_rec", np.mean(loc_rec_lst))
        # print("loc_pre", np.mean(loc_pre_lst))

        print("=" * 100)
        print("MISC F1 is ", np.mean(misc_f1_lst))
        # print("misc_rec", np.mean(misc_rec_lst))
        # print("misc_pre", np.mean(misc_pre_lst))

        # print("org_correct_lst", org_correct_lst)
        # print("total_correct  ", total_correct)

        print("="*100)

        print("org_correct_answer_lst", np.sum(org_correct_answer_lst))
        print("org_answer_length_lst", np.sum(org_answer_length_lst))
        print("org_predict_length_lst", np.sum(org_predict_length_lst))
        print("TP is", np.sum(org_correct_answer_lst))
        print("FN is", np.sum(org_answer_length_lst)-np.sum(org_correct_answer_lst))
        print("FP is", np.sum(org_predict_length_lst)-np.sum(org_correct_answer_lst))

        print("="*100)
        print("per_correct_answer_lst", np.sum(per_correct_answer_lst))
        print("per_answer_length_lst", np.sum(per_answer_length_lst))
        print("per_predict_length_lst", np.sum(per_predict_length_lst))
        print("TP is", np.sum(per_correct_answer_lst))
        print("FN is", np.sum(per_answer_length_lst)-np.sum(per_correct_answer_lst))
        print("FP is", np.sum(per_predict_length_lst)-np.sum(per_correct_answer_lst))

        print("="*100)
        print("loc_correct_answer_lst", np.sum(loc_correct_answer_lst))
        print("loc_answer_length_lst", np.sum(loc_answer_length_lst))
        print("loc_predict_length_lst", np.sum(loc_predict_length_lst))
        print("TP is", np.sum(loc_correct_answer_lst))
        print("FN is", np.sum(loc_answer_length_lst)-np.sum(loc_correct_answer_lst))
        print("FP is", np.sum(loc_predict_length_lst)-np.sum(loc_correct_answer_lst))

        print("="*100)
        print("misc_correct_answer_lst", np.sum(misc_correct_answer_lst))
        print("misc_answer_length_lst", np.sum(misc_answer_length_lst))
        print("misc_predict_length_lst", np.sum(misc_predict_length_lst))
        print("TP is", np.sum(misc_correct_answer_lst))
        print("FN is", np.sum(misc_answer_length_lst)-np.sum(misc_correct_answer_lst))
        print("FP is", np.sum(misc_predict_length_lst)-np.sum(misc_correct_answer_lst))

        categories = ["ALL", "ORG", "PER", "LOC", "MISC"]
        values = gpt_score_lst

        plt.bar(categories, values)
        plt.xlabel('Values')
        plt.ylabel('F1 Score')
        plt.title('NER')
        plt.show()

        return gpt_score_lst

    def compare_plot(self):
        gpt_score_lst = self.gpt_plot()
        human_score_lst = self.huamn_plot()

        # Data for the first plot
        categories_gpt = ["ALL", "ORG", "PER", "LOC", "MISC"]
        gpt_score_lst = gpt_score_lst

        # Data for the second plot
        categories_mean = ["ALL", "ORG", "PER", "LOC", "MISC"]
        score_lst = human_score_lst

        # Width of the bars
        bar_width = 0.35

        # Create a bar plot for GPT data

        plt.bar(categories_gpt, score_lst, width=bar_width, label='Human')
        # Adjust the positions for the bars of mean data
        categories_mean_adjusted = [x + bar_width for x in range(len(categories_mean))]

        plt.bar(categories_mean_adjusted, gpt_score_lst, width=bar_width, label='ChatGPT')

        # Set labels and title
        plt.xlabel('Values')
        plt.ylabel('F1 Score')
        plt.title('NER Comparison')
        plt.legend()  # Add legend to differentiate between GPT and Mean

        # Show the plot
        plt.show()






