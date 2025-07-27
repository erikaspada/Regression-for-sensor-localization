# Sensor Localization Pipeline  
Repository for the project 'Position detection of particles using RSD' assigned during the course Data Science Lab.

**Note:** The datasets (`development.csv`, `evaluation.csv`, `output.csv`) are not included due to privacy restrictions.

---

## Project Description

This project implements a machine learning pipeline to estimate the spatial position (x, y coordinates) of events based on signals recorded by a sensor network. The pipeline includes:

- Data preprocessing and cleaning (removal of noisy sensors, outliers, and data correction)  
- Feature engineering by deriving new features (e.g., `range[i]`)  
- Feature selection based on tree-based importance and correlation analysis  
- Multi-output regression model training  
- Spatial data visualization and feature analysis  

---

## Repository Structure
The folder `scr` contains Python modules organized by functionality:  
  - preprocessing.py  
  - visualization.py  
  - model.py  
  - prediction.py


## Requirements
See `requirements.txt` for Python packages.

## Authors
The authors of the project are:
- Giovanni Monco s315001@studenti.polito.it 
- Erika Spada s318375@studenti.polito.it 

