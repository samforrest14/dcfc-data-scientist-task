# Derby County FC – Data Scientist Technical Task

An analysis and lightweight decision-support tool exploring shot creation patterns using StatsBomb 360 data.

This repository contains my submission for the Derby County Football Club Data Scientist technical task.

## Project Overview

I tackled the **tactical patterns** aspect of the brief.

Using StatsBomb event and 360 data from the Bundesliga 2023/24 season, I investigated:

> What differentiates Bayer Leverkusen’s final-third passes that lead directly to shots from those that do not?

### Intended Audience
First-team coaching staff and performance analysts.

### Decision Informed
How attacking patterns should be designed and reviewed in order to improve direct shot creation from final-third passes.

## Repository Structure

- `project_outputs/` – main analytical notebook and 1-page written summary PDF
- `streamlit_app/` – lightweight Streamlit relationship visualisation tool
- `requirements.txt` – dependencies for reproduction

## Main Deliverables

- **Notebook:** `project_outputs/dcfc_data_science_task.ipynb`
- **1-page summary:** `project_outputs/DCFC_Data_Scientist_Task_Summary.pdf`
- **Dashboard:** Streamlit app in `streamlit_app/app.py`

## Key Findings

- Pass destination is the strongest driver of direct shot creation.
- Passes into the box are most likely to lead directly to a shot.
- Central areas outside the box are also valuable, but less so than the box.
- Defensive context matters: direct shot creation is most likely in dangerous congested areas, once this has been controlled for, pressure from defenders can negatively impact shot creation.
- The destination of the pass, and the defensive context it enters, matter more than the origin of the pass or the exact mechanics of its execution.

## Running the Notebook

Open the notebook in Jupyter:

```bash
jupyter lab
```

Then open:
`project_outputs/dcfc_data_science_task.ipynb`

## Running the Streamlit App

From the repository root, run in Python:
```bash
python -m streamlit run streamlit_app/app.py
```

The app allows users to simulate different passing scenarios and understand how pass destination, defensive context, and delivery type influence the likelihood of direct shot creation.

I have also deployed the Streamlit App using Streamlit Cloud so it does not need to be run locally. It can be accessed at the following link: https://dcfc-data-scientist-task-qyde5acdq2euwjb75we3fy.streamlit.app/

## Model interpretation
The final logistic regression model is intended as an interpretive tool rather than a predictive model.

Due to class imbalance and the strict definition of shot-assists, the model is most useful for understanding the direction and relative importance of key drivers of shot creation, rather than assigning high-confidence probabilities to individual passes.