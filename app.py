import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Career Path Predictor", page_icon="🎯")

st.title("🎯 Career Path Prediction System")
st.write("Fill in your details and click Predict to see your recommended career.")
st.markdown("---")

with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

career_courses = {
    'Applications Developer':                      ['Python Programming - Coursera', 'App Development - Udemy', 'Software Design - edX'],
    'CRM Technical Developer':                     ['Salesforce CRM - Trailhead (Free)', 'CRM Fundamentals - Coursera', 'Business Process - edX'],
    'Database Developer':                          ['SQL for Beginners - Khan Academy', 'MySQL / PostgreSQL - Udemy', 'Database Design - Coursera'],
    'Mobile Applications Developer':               ['Flutter - Udemy', 'React Native - Coursera', 'iOS with Swift - Apple Developer'],
    'Network Security Engineer':                   ['CompTIA Security+ - Udemy', 'Ethical Hacking - EC-Council', 'Networking - Cisco NetAcad'],
    'Software Developer':                          ['Python Full Course - Coursera', 'DSA - LeetCode', 'Clean Code - Udemy'],
    'Software Engineer':                           ['CS50 by Harvard - edX (Free)', 'System Design - Educative.io', 'OS - MIT OpenCourseWare'],
    'Software Quality Assurance (QA) / Testing':   ['ISTQB Certification', 'Selenium - Udemy', 'Agile & Scrum - Coursera'],
    'Systems Security Administrator':              ['CEH - EC-Council', 'Linux Security - Linux Foundation', 'CISSP - ISC2'],
    'Technical Support':                           ['Google IT Support - Coursera (Free)', 'CompTIA A+ - Udemy', 'Customer Service - LinkedIn Learning'],
    'UX Designer':                                 ['Google UX Design - Coursera', 'Figma UI/UX - Udemy', 'HCI - edX'],
    'Web Developer':                               ['Web Developer Bootcamp - Udemy', 'React JS - Scrimba', 'Full Stack Open - Helsinki (Free)'],
}

st.subheader("Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    logical_rating       = st.slider("Logical Quotient Rating", 1, 10, 5)
    hackathons           = st.slider("Hackathons Attended", 0, 10, 2)
    coding_rating        = st.slider("Coding Skills Rating", 1, 10, 5)
    public_speaking      = st.slider("Public Speaking Points", 1, 10, 5)
    self_learning        = st.selectbox("Self-Learning Capability?", ['yes', 'no'])
    extra_courses        = st.selectbox("Extra Courses Done?", ['yes', 'no'])
    team_work            = st.selectbox("Worked in Teams?", ['yes', 'no'])
    introvert            = st.selectbox("Are You Introverted?", ['yes', 'no'])
    senior_inputs        = st.selectbox("Taken Inputs from Seniors?", ['yes', 'no'])

with col2:
    reading_writing      = st.selectbox("Reading & Writing Skills", ['poor', 'medium', 'excellent'])
    memory_score         = st.selectbox("Memory Capability Score", ['poor', 'medium', 'excellent'])
    management_technical = st.selectbox("Management or Technical?", ['Management', 'Technical'])
    worker_type          = st.selectbox("Hard or Smart Worker?", ['hard worker', 'smart worker'])
    certifications       = st.selectbox("Certifications", ['information security', 'shell programming', 'r programming', 'distro making', 'machine learning', 'full stack', 'hadoop', 'app development', 'python'])
    workshops            = st.selectbox("Workshops", ['testing', 'database security', 'game development', 'data science', 'system designing', 'hacking', 'cloud computing', 'web technologies'])
    interested_subjects  = st.selectbox("Interested Subjects", ['programming', 'Management', 'data engineering', 'networks', 'Software Engineering', 'cloud computing', 'parallel computing', 'IOT', 'Computer Architecture', 'hacking'])
    career_area          = st.selectbox("Interested Career Area", ['testing', 'system developer', 'Business process analyst', 'security', 'developer', 'cloud computing'])
    company_type         = st.selectbox("Company Type", ['BPA', 'Cloud Services', 'product development', 'Testing and Maintainance Services', 'SAaS services', 'Web Services', 'Finance', 'Sales and Marketing', 'Product based', 'Service Based'])
    book_type            = st.selectbox("Interested Books", ['Series', 'Autobiographies', 'Travel', 'Guide', 'Health', 'Journals', 'Anthology', 'Dictionaries', 'Prayer books', 'Art'])

st.markdown("---")

if st.button("🔍 Predict Career", use_container_width=True):

    input_data = {
        'Logical quotient rating':            logical_rating,
        'hackathons':                         hackathons,
        'coding skills rating':               coding_rating,
        'public speaking points':             public_speaking,
        'self-learning capability?':          self_learning,
        'Extra-courses did':                  extra_courses,
        'certifications':                     certifications,
        'workshops':                          workshops,
        'reading and writing skills':         reading_writing,
        'memory capability score':            memory_score,
        'Interested subjects':                interested_subjects,
        'interested career area ':            career_area,
        'Type of company want to settle in?': company_type,
        'Taken inputs from seniors or elders':senior_inputs,
        'Interested Type of Books':           book_type,
        'Management or Technical':            management_technical,
        'hard/smart worker':                  worker_type,
        'worked in teams ever?':              team_work,
        'Introvert':                          introvert,
    }

    input_df = pd.DataFrame([input_data])

    for col in input_df.columns:
        if col in label_encoders:
            le = label_encoders[col]
            try:
                input_df[col] = le.transform(input_df[col])
            except:
                input_df[col] = 0

    input_array = input_df[feature_columns].to_numpy(dtype=float)

    prediction    = model.predict(input_array)[0]
    probabilities = model.predict_proba(input_array)[0]

    career_le        = label_encoders['Suggested Job Role']
    predicted_career = career_le.inverse_transform([prediction])[0]

    top3_idx = np.argsort(probabilities)[::-1][:3]
    top3     = [(career_le.inverse_transform([i])[0], round(probabilities[i] * 100, 1)) for i in top3_idx]

    st.success(f"### 🎉 Recommended Career: {predicted_career}")

    st.markdown("#### Top 3 Career Matches")
    medals = ["🥇", "🥈", "🥉"]
    for i, (career, prob) in enumerate(top3):
        st.write(f"{medals[i]} **{career}** — {prob}%")
        st.progress(int(prob))

    st.markdown("---")
    st.markdown(f"#### 📚 Recommended Courses for {predicted_career}")
    for course in career_courses.get(predicted_career, []):
        st.write("•", course)

st.caption("Career Path Prediction | GPI Internship — Cloud Counselage")
