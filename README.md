# CrowdsourcingVSLLM  

Use the project conda environment:
```python
conda env create -f environment.yml
```
Run Mathematical Reasoning experiment.
```python
python main.py --task math
```
Run Sentimental Analysis experiment.
```python
python main.py --task imdb
```
Run Named Entity Recognition experiment.
```python
python main.py --task ner
```
Run Persuasiveness experiment.
```python
python main.py --task persuasiveness
```
Our repo contains all the data files we used in this project to re-produce our results. If you want to re-run ChatGPT, you should replace the OPENAI key by yourself.  

The follows are our survey in Qualtrics platform:

 - [Mathematical Reasoning](https://wustl.az1.qualtrics.com/jfe/preview/previewId/ad85437d-8fb5-4edd-9da5-f841a41b7ecb/SV_bqhlLFavcvhqxgi?Q_CHL=preview&Q_SurveyVersionID=current)
 - [Sentimental Analysis](https://wustl.az1.qualtrics.com/jfe/preview/previewId/485ebaf8-efe1-46bb-8f64-7014cadea47d/SV_6lhsqVB7KvJL7rU?Q_CHL=preview&Q_SurveyVersionID=current)
 - [Named Entity Recognition](https://wustl.az1.qualtrics.com/jfe/preview/previewId/42ca64ab-0310-41bf-9b45-58799245c6ba/SV_6u8cTjlGPqq9ZDo?Q_CHL=preview&Q_SurveyVersionID=current)
 - [Persuasiveness](https://wustl.az1.qualtrics.com/jfe/preview/previewId/853d084e-b6b1-4b4b-86fc-00478011372c/SV_3ghHAlXQgEDJ2hE?Q_CHL=preview&Q_SurveyVersionID=current)
