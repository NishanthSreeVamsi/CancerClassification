from flask import Flask, render_template, request
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import numpy as np
app = Flask('__name__')
model=pickle.load(open('model.pkl','rb'))

def hist(file):
    df = file
    plt.figure(figsize=(14,6))
    plt.hist(df['Class'])
    plt.xlabel('Type of Cancer')
    plt.ylabel('Samples')
    return plt


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/classification')
def classification():
    return render_template('classification.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')


@app.route('/geneanalysis', methods=['POST'])
def geneanalysis():
    file = request.files['filean']
    if file:
        #Reading file
        df = pd.read_csv(file)
        label = pd.read_csv("Data/labels.csv")
        #Generating Histogram
        histogrm = hist(df)
        histogrm.savefig('static/histogrm.png')
        #Dataset for PCA
        df_pca = df.drop(['Unnamed: 0'], axis=1)
        df_pca = df_pca.drop(['Class'], axis=1)
        x_pca = df_pca.values
        #Scaling process
        scaler = StandardScaler()
        X_Scaled = scaler.fit_transform(x_pca)
        #Performing PCA with .995 variance
        pca_with_995=PCA(n_components=2)
        X_pca_with_995 = pca_with_995.fit_transform(x_pca)
        #Defining df_cat_data categorical data
        df_cat_data = df
        df_cat_data['Class'] = df_cat_data['Class'].map({'PRAD': 1, 'LUAD': 2, 'BRCA': 3, 'KIRC': 4, 'COAD': 5}) 
        df_cat_data = df_cat_data.drop(['Unnamed: 0'],axis=1)
        
        #Generating Scatter plot
        df_pca_995 = pd.DataFrame(X_pca_with_995)
        df_pca_995['cancer_type']=df_cat_data['Class']

        # Add the convereted categorical data for 
        
        plt.figure(figsize=(4,4))
        scatter_plot = sns.scatterplot(x=0,y=1,hue = 'cancer_type', data=df_pca_995)
        
        # Set the current figure to the scatter plot
        fig = scatter_plot.get_figure()

        # Save the plot as a file
        plot_path = 'static/scatter_plot.png'
        fig.savefig(plot_path)
        plt.close(fig)
        
        #Part 2 Clustering
        
        clusters_995 = KMeans(5, n_init = 5)
        clusters_995.fit(X_pca_with_995)
        pca_with_995_data_frame = pd.DataFrame(data=X_pca_with_995)
        pca_with_995_data_frame['Cls_label'] = clusters_995.labels_
        pca_with_995_data_frame['given_cancer_type'] = label.Class.values
        brca_995 = pca_with_995_data_frame.groupby('given_cancer_type').get_group('BRCA')
        value_counts = brca_995.Cls_label.value_counts()
        luad_995 = pca_with_995_data_frame.groupby('given_cancer_type').get_group('LUAD')
        value_counts2 = luad_995.Cls_label.value_counts()
        coad_995 = pca_with_995_data_frame.groupby('given_cancer_type').get_group('COAD')
        value_counts3 = coad_995.Cls_label.value_counts()
        prad_995 = pca_with_995_data_frame.groupby('given_cancer_type').get_group('PRAD')
        value_counts4 = prad_995.Cls_label.value_counts()
        kirc_995 = pca_with_995_data_frame.groupby('given_cancer_type').get_group('KIRC')
        value_counts5 = kirc_995.Cls_label.value_counts()
        kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit_predict(X_pca_with_995)
        plt.figure()
        plt.scatter(X_pca_with_995[:,0], X_pca_with_995[:,1])
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
        plt.savefig('static/scat1.png')
        plt.close('all')
        return render_template('table2.html',  value_counts5=value_counts5,value_counts4=value_counts4, value_counts3=value_counts3,  value_counts=value_counts,  value_counts2=value_counts2)
    else:
        return "No file selected."
    



@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        df = pd.read_csv(file)
        newdf=df.tail(100)
        sample = newdf.iloc[:, 0].values
        df_lda = newdf.drop(['Unnamed: 0'], axis=1)
        x_lda = df_lda
        prediction=model.predict(x_lda)
        df = pd.DataFrame(prediction)
        df.columns = ['Type']
        
        df['Sample'] = sample
        df['Type'] = df['Type'].map({ 1 : 'PRAD', 2 : 'LUAD', 3 : 'BRCA', 4 : 'KIRC', 5 : 'COAD'})
        
        return render_template('classresult.html', table=df.to_html(index=False))
    else:
        return "No file selected."

if __name__ == '__main__':
    app.run(debug=True)


