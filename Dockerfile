FROM python:3.7

RUN pip install numpy==1.18.1 matplotlib==3.2.0 scikit-learn==0.22.2.post1 click==7.0

COPY scripts /fun_with_ml