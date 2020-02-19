from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import tokenize
import joblib
import pandas as pd

app = Flask(__name__)
@app.route('/')
def func_01():
    return render_template('home.html')

@app.route('/jobs', methods=['POST'])
def func_02():
    # return render_template
    if request.method == 'POST':
        data = request.form
        # skills = [' '.join([data['skills'], data['experiences'], data['education']])]
        skills=[data['skills']]
        job_category = model.predict(skills)[0]
        df_job_field = df[df['Category'] == job_category]
        job_category_qualification = df[df['Category'] == job_category]['Qualifications'].tolist()
        analyze = skills + job_category_qualification
        count_vectorize = CountVectorizer(stop_words='english')
        matrix = count_vectorize.fit_transform(analyze)
        cos_similarity = cosine_similarity(matrix)
        job_list = sorted(list(enumerate(cos_similarity[0])), key=lambda x:x[1], reverse=True)
        print(job_list)
        print(job_list[1][1])
        if job_list[1][1] <= 0.1:
            return "no job match for you"
        else:
            idx = []
            for i,_ in job_list[1:6]:
                idx.append(i-1)
            idx.sort()
            data_final = df_job_field.iloc[idx]['Title'].tolist()
            responsibility = df_job_field.iloc[idx]['Responsibilities'].tolist()
            data_minimum_requirement = df_job_field.iloc[idx]['Minimum_Qualifications'].tolist()
            data_preferred_requirement = df_job_field.iloc[idx]['Preferred_Qualifications'].tolist()
            job_loc = df_job_field.iloc[idx]['Location'].tolist()
            # print(qualifications)
            return render_template('jobs.html', data=zip(data_final, data_minimum_requirement, data_preferred_requirement, responsibility, job_loc), job_cat=job_category)
    else:
        return "The method is not POST"

if __name__ == "__main__":
    model = joblib.load('linearsvc_model.sav')
    df = pd.read_csv('./dataset/job/job_skills_clean_final.csv')
    app.run(debug=True)