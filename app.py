import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# LUNG
data_lung=pd.read_csv('lung_final.csv')
x_l=data_lung[['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
       'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',
       'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
       'SWALLOWING DIFFICULTY', 'CHEST PAIN']]
y_l=data_lung['LUNG_CANCER']

Xl_train,xl_test,Yl_train,yl_test=train_test_split(x_l,y_l,test_size=0.2)
model_forest_lung=RandomForestClassifier()
model_forest_lung.fit(Xl_train,Yl_train)

# HEART

data_heart=pd.read_csv('heart_final.csv')
x_h=data_heart[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
       'ejection_fraction', 'high_blood_pressure', 'platelets',
       'serum_creatinine', 'serum_sodium', 'sex', 'smoking']]
y_h=data_heart['DEATH_EVENT']

Xh_train,xh_test,Yh_train,yh_test=train_test_split(x_h,y_h,test_size=0.2)
model_forest_heart=RandomForestClassifier()
model_forest_heart.fit(Xh_train,Yh_train)

# BRAIN
data_brain=pd.read_csv('brain_final.csv')
x_b=data_brain[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status']]
y_b=data_brain['stroke']
Xb_train,xb_test,Yb_train,yb_test=train_test_split(x_b,y_b,test_size=0.2)

model_forest_brain=RandomForestClassifier()
model_forest_brain.fit(Xb_train,Yb_train)


with st.sidebar:
    st.write('## 🩺Choose what to Diagnose for:')
    services = st.radio(
        "",
        (" 🧠Brain Stroke Detection", " 🫁Lung Cancer Detection"," 🫀Heart Failure Detection")
    )

if services==" 🧠Brain Stroke Detection":
    # gender,age,hypertension,heart_disease,ever_married,work_type
    # ,Residence_type,avg_glucose_level,bmi,smoking_status,stroke
    st.warning(' ')
    st.title("Brain Stroke Detection 🧠")
    col1,col2=st.columns(2)
    with col1:
        st.image('Brainstroke.jpg')
    with col2:
        st.subheader("About")
        st.write("➡ Brain stroke  is a medical condition where the blood supply to a portion of the brain decreases or gets severely interrupted. It is a medical emergency wherein the cells of the brain start dying within minutes of being deprived of nutrients and oxygen due to the restriction of blood supply.")
        st.info("➡ Symptoms of brain stroke include, Blurred, blackened or double vision in one or both eyes,Difficulty in speaking, slurring of speech and confusion, Difficulty in walking and balancing and Sudden and one-sided paralysis or numbness of an arm or leg and face. ")
    st.error('')
    st.subheader("👩🏻‍⚕️ Let's Predict ")
    st.info('Fill below information')
    col1,col2,col3=st.columns(3)
    with col1:
        g1=st.selectbox('Specify your Gender',("","Male","Female"))
        g_1=g1
        if(g1=="Male"):
            g1=0
        if(g1=="Female"):
            g1=1
        
        age1= st.number_input('Enter your age')
       
        hyper_ten=st.selectbox('Do you suffer with hypertension',("",'Yes','No'))
        
        h_t=hyper_ten
        
        if(hyper_ten=='Yes'):
            hyper_ten=1
        if(hyper_ten=='No'):
            hyper_ten=0
    with col2:
        h_dis=st.selectbox('Do you suffer from heart disease',('','Yes','No'))
        h_d=h_dis
        if(h_dis=='Yes'):
            h_dis=1
        if(h_dis=='No'):
            h_dis=0
        
        r_type=st.selectbox('Select your Residence type',('','Urban','Rural'))
        r_t=r_type
        if(r_type=='Urban'):
            r_type=1
        if(r_type=='Rural'):
            r_type=0
        
        av_glucose=st.number_input('Enter your Average glucose level')
    
    with col3:
        mar1=st.selectbox('Are you married',('','Yes','No'))
        mm_1=mar1
        if(mar1=='Yes'):
            mar1=1
        if(mar1=='No'):
            mar1=0
        
        work_type1=st.selectbox('What is your professional status',('','Private','Self-employed','Goverment job','Student'))
        w_t_1=work_type1
        if(work_type1=='Private'):
            work_type1=0
        if(work_type1=='Self-employed'):
            work_type1=1
        if(work_type1=='Goverment job'):
            work_type1=2
        if(work_type1=='Student'):
            work_type1=3
        
        bmi=st.number_input("Enter your Body Mass Index (bmi)")
    smoke1=st.selectbox('Do you Smoke',('','Formerly Smoker','Never Smoke','Smoker','Do not want to specify'))
    s1ke=smoke1
    if(smoke1=='Formerly Smoker'):
        smoke1=0
    if(smoke1=='Never Smoke'):
        smoke1=1
    if(smoke1=='Smoker'):
        smoke1=2
    if(smoke1=='Do not want to specify'):
        smoke1=3
    
    if(st.button('Predict')):
        inp_brain=[[g1,age1,hyper_ten,h_dis,mar1,work_type1,r_type,av_glucose,bmi,smoke1]]
        if(model_forest_brain.predict(inp_brain)==1):
            st.info('')
            st.subheader('📑Patient report')
            c1,c2=st.columns(2)
            with c1:
                st.write('**Gender**')
                st.write('**Age**')
                st.write('**Hypertension**')
                st.write('**Heart Disease**')
                st.write('**Residence type**')
                st.write('**Average Glucose levels**')
                st.write('**Married**')
                st.write('**Work type**')
                st.write('**BMI**')
                st.write('**Smoking status**')
                st.write("")
                st.write("Final Outcome")
            with c2:
                st.write(g_1)
                st.write(age1)
                st.write(h_t)
                st.write(h_d)
                st.write(r_t)
                st.write(av_glucose)
                st.write(mm_1)
                st.write(w_t_1)
                st.write(bmi)
                st.write(s1ke)
                st.write("")
                st.success("Likely to get Brain stroke")
        if(model_forest_brain.predict(inp_brain)==0):
            st.info(" ")
            st.subheader('📑Patient report')
            c1,c2=st.columns(2)
            with c1:
                st.write('**Gender**')
                st.write('**Age**')
                st.write('**Hypertension**')
                st.write('**Heart Disease**')
                st.write('**Residence type**')
                st.write('**Average Glucose levels**')
                st.write('**Married**')
                st.write('**Work type**')
                st.write('**BMI**')
                st.write('**Smoking status**')
                st.write("")
                st.write("**Final Outcome**")
            with c2:
                st.write(g_1)
                st.write(age1)
                st.write(h_t)
                st.write(h_d)
                st.write(r_t)
                st.write(av_glucose)
                st.write(mm_1)
                st.write(w_t_1)
                st.write(bmi)
                st.write(s1ke)
                st.write("")
                st.error("Unlikely to get Brain stroke ")
    st.warning('')

if services==" 🫁Lung Cancer Detection":
    st.warning(' ')
    st.title("Lung Cancer Detection 🫁")
    col_1,col_2=st.columns(2)
    with col_1:
        st.image('lungcancer.jpg')
    with col_2:
        st.subheader(" About")
        st.write("➡ Lung cancer happens when cells in the lung change (or mutate). Most often, this is because of exposure to dangerous chemicals that we breathe ")
        st.write("➡ Lung cancer affects the respiratory system, but it can spread to distant areas and many of the body’s systems.")
        st.info("➡ Symptoms of lung cancer can vary from patient to patient. For some, it may feel like a persistent cough or respiratory infection, shortness of breath, or shoulder, arm, chest, or back pain.")
    st.error('')
    st.subheader("👩🏻‍⚕️ Let's Predict ")
    st.info('Fill below information')
    col_l_1,col_l_2,col_l_3=st.columns(3)
    with col_l_1:
        sex_m_f=st.selectbox('Select your Gender',('','Male','Female'))
        s_mf=sex_m_f
        if(sex_m_f=='Male'):
            sex_m_f=0
        if(sex_m_f=='Female'):
            sex_m_f=1
        
        smoke2=st.selectbox('Do you Smoke',('','Never Smoke','Smoker'))
        sk2=smoke2
        if(smoke2=='Never Smoke'):
            smoke2=1
        if(smoke2=='Smoker'):
            smoke2=2
        
        age_2= st.number_input('Your age input')
        
        y_fing=st.selectbox('Do you suffer from Yellow Fingers',('','Yes','No'))
        y_f=y_fing
        if(y_fing=='Yes'):
            y_fing=2
        if(y_fing=='No'):
            y_fing=1

        anx=st.selectbox('Do you Suffer from Anxiety',('','Yes','No'))
        a_x=anx
        if(anx=='Yes'):
            anx=2
        if(anx=='No'):
            anx=1
    with col_l_2:
        
        p_press=st.selectbox('Do you suffer from Peer Pressure',('','Yes','No')) 
        p_prs=p_press
        if(p_press=='Yes'):
            p_press=2
        if(p_press=='No'):
            p_press=1
        
        c_dis=st.selectbox('Do you suffer from Chronic diseases',('','Yes','No'))
        c_disis=c_dis
        if(c_dis=='Yes'):
            c_dis=2
        if(c_dis=='No'):
            c_dis=1
        
        fatig=st.selectbox('Do you feel Fatigue',('','Very often','Sometimes'))
        fg=fatig
        if(fatig=='Very often'):
            fatig=2
        if(fatig=='Sometimes'):
            fatig=1
        
        allgy=st.selectbox('Are you Allergic',('','Yes','No'))
        algy=allgy
        if(allgy=='Yes'):
            allgy=2
        if(allgy=='No'):
            allgy=1
        
        whe=st.selectbox('Do you suffer from Wheezing',('','Yes','No'))
        wheee=whe
        if(whe=='Yes'):
            whe=2
        if(whe=='No'):
            whe=1
    with col_l_3:
        alch=st.selectbox('Do you consume Alcohol',('','Yes','No'))
        al_ol=alch
        if(alch=='Yes'):
            alch=2
        if(alch=='No'):
            alch=1
        
        cough=st.selectbox('Do you suffer from Coughing',('','Yes','No'))
        cghh=cough
        if(cough=='Yes'):
            cough=2
        if(cough=='No'):
            cough=1
        
        shrt_b=st.selectbox('Suffering from Shortness of Breath',('','Yes','No'))
        s_b_r=shrt_b
        if(shrt_b=='Yes'):
            shrt_b=2
        if(shrt_b=='No'):
            shrt_b=1
        
        sw_dif=st.selectbox('Do you feel Difficulty in Swallowing',('','Yes','No'))
        s_w_d=sw_dif
        if(sw_dif=='Yes'):
            sw_dif=2
        if(sw_dif=='No'):
            sw_dif=1
        
        ch_pain=st.selectbox('Do you feel chest pain',('','Very often','Sometimes'))
        c_p_n=ch_pain
        if(ch_pain=='Very often'):
            ch_pain=2
        if(ch_pain=='Sometimes'):
            ch_pain=1
    if(st.button('Predict')):
        inp_lung_val=[[sex_m_f,age_2,smoke2,y_fing,anx,p_press,c_dis,fatig,allgy,whe,alch,cough,shrt_b,sw_dif,ch_pain]]
        if(model_forest_lung.predict(inp_lung_val)==1):
            st.info('')
            st.subheader('📑Patient report')
            c1,c2=st.columns(2)
            with c1:
                    st.write('**Gender**')
                    st.write('**Age**')
                    st.write('**Yellow finger**')
                    st.write('**Anxiety status**')
                    st.write('**Peer Pressure**')
                    st.write('**Chronic Disease**')
                    st.write('**Fatigueness Status**')
                    st.write('**Allergic**')
                    st.write('**Wheezing status**')
                    st.write('**Smoking status**')   
                    st.write("**Consumption of Alcohol**")
                    st.write("**Coughing status**")
                    st.write("**Shortness of breath**")
                    st.write("**Difficulty in Swallowing**")
                    st.write("**Chest pain**")
                    st.write("")
                    st.write("**Final Outcome**")

            with c2:
                    st.write(s_mf)
                    st.write(age_2)
                    st.write(y_f)
                    st.write(a_x)
                    st.write(p_prs)
                    st.write(c_disis)
                    st.write(fg)
                    st.write(algy)
                    st.write(wheee)
                    st.write(sk2)
                    st.write(al_ol)
                    st.write(cghh)
                    st.write(s_b_r)
                    st.write(s_w_d)
                    st.write(c_p_n)
                    st.write("")
                    st.success("Likely to have Lung Cancer")
        if(model_forest_lung.predict(inp_lung_val)==0):
            st.info('')
            st.subheader('📑Patient report')
            c1,c2=st.columns(2)
            with c1:
                    st.write('**Gender**')
                    st.write('**Age**')
                    st.write('**Yellow finger**')
                    st.write('**Anxiety status**')
                    st.write('**Peer Pressure**')
                    st.write('**Chronic Disease**')
                    st.write('**Fatigueness Status**')
                    st.write('**Allergic**')
                    st.write('**Wheezing status**')
                    st.write('**Smoking status**')   
                    st.write("**Consumption of Alcohol**")
                    st.write("**Coughing status**")
                    st.write("**Shortness of breath**")
                    st.write("**Difficulty in Swallowing**")
                    st.write("**Chest pain**")
                    st.write("")
                    st.write("**Final Outcome**")

            with c2:
                    st.write(s_mf)
                    st.write(age_2)
                    st.write(y_f)
                    st.write(a_x)
                    st.write(p_prs)
                    st.write(c_disis)
                    st.write(fg)
                    st.write(algy)
                    st.write(wheee)
                    st.write(sk2)
                    st.write(al_ol)
                    st.write(cghh)
                    st.write(s_b_r)
                    st.write(s_w_d)
                    st.write(c_p_n)
                    st.write("")
                    st.error("Unlikely to have Lung Cancer")
 
    st.warning(' ')


if services==" 🫀Heart Failure Detection":

    st.warning(' ') 
    st.title("Heart Failure Detection 🫀")
    colh1,colh2=st.columns(2)
    with colh1:
        st.image('heartfail .png')
    with colh2:
        st.subheader(" About")
        st.write("➡ Heart failure is a progressive condition in which the heart's muscle  gets injured from something like a heart attack or high blood pressure and gradually loses its ability to pump enough blood to supply the body's needs. ")
        st.write("➡ The heart can be affected in two ways, either become weak and unable to pump blood  or it become stiff and unable to fill with blood adequately. ")
        st.info("➡  Symptoms of Heart failure include, Shortness of breath with activity or when lying down,Rapid or irregular heartbeat and many more.")
    st.error('')
    st.subheader("👩🏻‍⚕️ Let's Predict ")
    st.info('Fill below information')
    col_h_1,col_h_2,col_h_3=st.columns(3)
    with col_h_1:
        age_e=st.number_input("Enter Your Age")
        
        sex=st.selectbox("Select your Gender",("","Male","Female"))
        s_xxx=sex
        if(sex=="Male"):
            sex=1    
        if(sex=="Female"):
            sex=0
        
        an=st.selectbox('Do you suffer from Anaemia',('','Yes','No'))
        anmi=an
        if(an=='Yes'):
            an=1
        if(an=='No'):
            an=0
    with col_h_2:
        cv=st.number_input('Enter your creatinine phosphokinase levels')
        
        db=st.selectbox('Are you Diabetic',('','Yes','No'))
        dbt_cc=db
        if(db=='Yes'):
            db=1
        if(db=='No'):
            db=0
        
        ef_frac=st.number_input("Enter your ejection fraction levels")
    with col_h_3:
        hb_pres=st.selectbox('Do you have high blood pressure',('','Yes','No'))
        h_b_prss=hb_pres
        if(hb_pres=='Yes'):
            hb_pres=1
        if(hb_pres=='No'):
            hb_pres=0
        
        s_cr=st.number_input('Enter your serum creatinine levels')
        
        s_sod=st.number_input('Enter your serum sodium levels')

    plt_ts=st.number_input('Enter your platelets count')
    smke=st.selectbox('Do you smoke',('','Yes','No'))
    sk3=smke
    if(smke=='Yes'):
        smke=1
    if(smke=='No'):
        smke=0
    if(st.button('Predict')):
        inp_heart=[[age_e,an,cv,db,ef_frac,hb_pres,plt_ts,s_cr,s_sod,sex,smke]]
        if(model_forest_heart.predict(inp_heart)==1):
            st.info('')
            st.subheader('📑Patient report')
            c1,c2=st.columns(2)
            with c1:
                    st.write('**Gender**')
                    st.write('**Age**')
                    st.write('**Anaemia status**')
                    st.write('**Creatine Phosphokinase levels**')
                    st.write('**Diabetic**')
                    st.write('**Ejection fraction levels**')
                    st.write('**High blood pressure**')
                    st.write('**Serum creatinine levels**')
                    st.write('**Serum sodium levels**')
                    st.write('**Smoking status**')   
                    st.write("**Platelets Count**")
                    st.write("")
                    st.write("**Final Outcome**")
            with c2:
                    st.write(s_xxx)
                    st.write(age_e)
                    st.write(anmi)
                    st.write(cv)
                    st.write(dbt_cc)
                    st.write(ef_frac)
                    st.write(h_b_prss)
                    st.write(s_cr)
                    st.write(s_sod)
                    st.write(sk3)
                    st.write(plt_ts)
                    st.write("")
                    st.success("Likely to have Heart failure")
        if(model_forest_heart.predict(inp_heart)==0):
            st.info('')
            st.subheader('📑Patient report')
            c1,c2=st.columns(2)
            with c1:
                    st.write('**Gender**')
                    st.write('**Age**')
                    st.write('**Anaemia status**')
                    st.write('**Creatine Phosphokinase levels**')
                    st.write('**Diabetic**')
                    st.write('**Ejection fraction levels**')
                    st.write('**High blood pressure**')
                    st.write('**Serum creatinine levels**')
                    st.write('**Serum sodium levels**')
                    st.write('**Smoking status**')   
                    st.write("**Platelets Count**")
                    st.write("")
                    st.write("**Final Outcome**")
            with c2:
                    st.write(s_xxx)
                    st.write(age_e)
                    st.write(anmi)
                    st.write(cv)
                    st.write(dbt_cc)
                    st.write(ef_frac)
                    st.write(h_b_prss)
                    st.write(s_cr)
                    st.write(s_sod)
                    st.write(sk3)
                    st.write(plt_ts)
                    st.write("")
                    st.error("Unlikely to have Heart failure")
    st.warning(' ')

# st.info('')
# st.subheader('📑Patient report')
# c1,c2=st.columns(2)
# with c1:
#     st.write('**Gender**')
#     st.write('**Age**')
#     st.write('**Anaemia status**')
#     st.write('**Creatine Phosphokinase levels**')
#     st.write('**Diabetic**')
#     st.write('**Ejection fraction levels**')
#     st.write('**High blood pressure**')
#     st.write('**Serum creatinine levels**')
#     st.write('**Serum sodium levels**')
#     st.write('**Smoking status**')   
#     st.write("**Platelets Count**")
#     st.write("")
#     st.write("Final Outcome")

# with c2:
#     st.write(s_xxx)
#     st.write(age_e)
#     st.write(anmi)
#     st.write(cv)
#     st.write(dbt_cc)
#     st.write(ef_frac)
#     st.write(h_b_prss)
#     st.write(s_cr)
#     st.write(s_sod)
#     st.write(sk3)
#     st.write(plt_ts)
#     st.write("")
#     st.success("Likely to have Heart failure")
