# Local SHAP and LIME Explanations for a Hybrid 1DCNN–BiLSTM–HybridAttention Model in Telecom Customer Churn Prediction.

## 🛠️ Getting Started

To run the code for this project, please follow the instructions below. 

> **Note:** Before you download and run the code, ensure you are working inside the `Source-code and dataset` folder.

### 1. Prerequisites
You will need an IDE to run this code. It is recommended to use **PyCharm IDE** (this project was developed using PyCharm Community Edition 2024.3.1.1). You only need to run the `main.py` module.

### 2. Install Dependencies
Open a terminal in the `Source-code and dataset` folder and install all the required Python packages listed in the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### 3. Dataset Requirement
Inside the `Source-code and dataset folder`, there is a Kaggle telecom churn dataset https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset?resource=download
named WA_Fn-UseC_-Telco-Customer-Churn.csv used for this research.
Ensure you download this dataset and place it in the same folder as the Python code files before running the pipeline.

### 4. Execution & GUI Interaction
* After opening the `Source-code and dataset` folder inside your PyCharm IDE, you just need to run the `main.py` module.
> **⚠️ Attention:** During the pipeline execution, a popup **GUI form** will display to allow you to select a churned customer's original data. 
> *   You must enter a customer index ranging from **0–1408** (from the test dataset) for local explanation, then click **OK**.
> *   *(Alternatively, you can use the pre-selected customer index **"34"** which represents a churned customer profile. Simply click **OK** directly).*
> *   Finally, click the **"Predict this customer"** button at the bottom-right section of this form to continue the prediction and explanation process.

<img width="975" height="571" alt="image" src="https://github.com/user-attachments/assets/940280ac-94d2-44c7-aa07-3906d722c232" />
<img width="975" height="574" alt="image" src="https://github.com/user-attachments/assets/70d07af5-334f-4e1a-b6f7-c0c58b3c556d" />

**Viewing the Results:**
After the running process is finished, if you have a default web browser on your computer, the result report will open automatically. 

* **Fallback:** If the browser does not open automatically, you will find the result generated in the same folder as an HTML file named `results_report.html`
                together with the python code files.
* You can open it using any web browser to see the final results of the pipeline.

### 📊 Pre-computed Results (Artifacts)

If you do not wish to run the code, pre-computed results from this ML pipeline are available in the Artifact folder. There are two versions:
1. `results_report.md`: A Markdown file that allows you to directly view the screen shot results of the pipeline within GitHub.
2. `results_report.html`: An interactive HTML file used for the final research (coursework).
   You can download and open this file in any web browser to see the full results of the pipeline.
