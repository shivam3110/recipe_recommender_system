conda create -y --name recipe python==3.12.5
conda activate recipe

conda install -y -c conda-forge transformers=4.49.0
conda install -y -c conda-forge pandas=2.2.3
conda install -y -c conda-forge hvplot=0.11.2
conda install -y -c conda-forge umap-learn=0.5.7
conda install -y -c conda-forge scikit-learn=1.6.1
conda install -y -c conda-forge matplotlib=3.10.1
conda install -y -c conda-forge ipykernel=6.29.4
python -m ipykernel install --user --name recipe --display-name="recipe"

pip3 install torch==2.6.0
pip install sentence_transformers==3.3.1
pip install plotly==6.0.1
pip install wordcloud==1.9.4