# AlignCheck
AlignCheck: a Semantic Open-Domain Metric for Factual Consistency Assessment

 
### Prerequisites

1. **Install required Python libraries:**

```bash
pip install -r requirements.txt
```

2. **Download SNOMED-CT model pack**  
- [MedCAT SNOMED-CT model](https://github.com/CogStack/MedCAT/blob/main/docs/main.md) (requires NIH authentication)  
- Set `medcat_path` in `main.py`.

3. **Download CoreNLP for OpenIE**  
- [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/download.html)  
- Set `CORE_NLP_PATH` in `main.py`.

4. **Format the source and target texts similar to sample_input.json**


5. **Run the scoring script:**

```bash
python main.py
```
