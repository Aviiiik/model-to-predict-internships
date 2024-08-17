import csv
import os
import time
from datetime import datetime
import matplotlib
import pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('Agg')
sns.set(rc={'figure.figsize': (8, 5)})

app = Flask(__name__)
locations = []
with open('Files/locations_list.txt') as f:
    for line in f:
        locations.append(line.strip())
locations.sort()
locations_values = ['location_' + x for x in locations]
locations.insert(0, 'Select Location')
locations_values.insert(0, '')
locations.insert(1, 'Find the best one')
locations_values.insert(1, 'Find')

skills = []
with open('Files/skills_list.txt') as f:
    for line in f:
        skills.append(line.strip())
skills.sort()
skills_values = ['skill_' + x for x in skills]
skills.insert(0, 'Select Skill')
skills_values.insert(0, '')

perks = []
with open('Files/perks_list.txt') as f:
    for line in f:
        perks.append(line.strip())
perks.sort()
perks_values = ['perk_' + x for x in perks]
perks.insert(0, 'Select Perks')
perks.insert(1, 'None')
perks_values.insert(0, '')
perks_values.insert(1, 'None')

categories = []
with open('Files/categories_list.txt') as f:
    for line in f:
        categories.append(line.strip())
categories.sort()
categories_values = ['category_' + x for x in categories]
categories.insert(0, 'Select Categories')
categories_values.insert(0, '')

file = open('Sq_model.pkl', 'rb')
model = pickle.load(file)
file2 = open("scaler.pkl", 'rb')
scaler = pickle.load(file2)
file3 = open('X_columns.pkl', 'rb')
X_columns = pickle.load(file3)


def write_log(name, email, locations, skills, perks, durations, categories):
    with open("customized_query.log", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now(), name[0], email[0], locations[0], skills, perks, durations, categories[0]])


def predict_values(name, email, locations, skills, perks, durations, categories):
    _ = {}
    for skill in skills:
        _.update({skill: 1})

    for location in locations:
        _.update({location: 1})

    for perk in perks:
        if perk != 'None':
            _.update({perk: 1})

    for category in categories:
        _.update({category: 1})

    _.update({'duration': durations})
    df = pd.DataFrame([_], columns=X_columns)
    df = df.fillna(0)
    prediction = round(np.square(model.predict(pd.DataFrame(scaler.transform(df), columns=X_columns)))[0])
    return prediction

#Test 1
print("Test Case 1")
print("Testing with various limiting values of duration while taking random values for other features")
print("Output with duration = 0. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],[skills_values[2],skills_values[13]],[perks_values[4],perks_values[3]],0,[categories_values[2]]))
print("Output with duration = 1. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],[skills_values[2],skills_values[13]],[perks_values[4],perks_values[3]],-6,[categories_values[2]]))
print("Output with duration = 36. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],[skills_values[2],skills_values[13]],[perks_values[4],perks_values[3]],36,[categories_values[2]]))
print("Output with duration = -36. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],[skills_values[2],skills_values[13]],[perks_values[4],perks_values[3]],-36,[categories_values[2]]))
print("Output with duration = 1000. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],[skills_values[2],skills_values[13]],[perks_values[4],perks_values[3]],1000,[categories_values[2]]))
print("Output with duration = -1000. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],[skills_values[2],skills_values[13]],[perks_values[4],perks_values[3]],-1000,[categories_values[2]]))
print("Output with duration = 1000000. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],[skills_values[2],skills_values[13]],[perks_values[4],perks_values[3]],1000000,[categories_values[2]]))
print("Output with duration = -1000000. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],[skills_values[2],skills_values[13]],[perks_values[4],perks_values[3]],-1000000,[categories_values[2]]))
print("Output with duration = 0.0000001. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],[skills_values[2],skills_values[13]],[perks_values[4],perks_values[3]],0.0000001,[categories_values[2]]))
print("Output with duration = -0.0000001. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],[skills_values[2],skills_values[13]],[perks_values[4],perks_values[3]],-0.0000001,[categories_values[2]]))


#Test 2
print("Test Case 2")
print("Testing with various different combination of skills while taking random values for other features")
print("Output with 3 same type of skills. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],["skill_Adobe Premiere Pro","skill_Adobe Photoshop Lightroom CC","skill_Adobe Photoshop"],[perks_values[1]],3,[categories_values[2]]))
print("Output with 3 different type of skills. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],["skill_Bengali Proficiency (Spoken)","skill_AutoCAD","skill_Arduino"],[perks_values[1]],3,[categories_values[2]]))


#Test 3
print("Test Case 3")
print("Testing with various different combination of perks while taking random values for other features")
print("Output with combination of perk3 and perk4. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],[skills_values[2],skills_values[13]],[perks_values[3],perks_values[4]],3,[categories_values[2]]))
print("Output with combination of perk1 and perk6. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],[skills_values[2],skills_values[13]],[perks_values[1],perks_values[5]],3,[categories_values[2]]))
print("Output with combination of perk5 and perk7. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],[skills_values[2],skills_values[13]],[perks_values[5],perks_values[6]],3,[categories_values[2]]))
print("Output with combination of perk2 and perk3. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],[skills_values[2],skills_values[13]],[perks_values[2],perks_values[3]],3,[categories_values[2]]))
print("Output with combination of perk4, perk1, and perk 6. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],[skills_values[2],skills_values[13]],[perks_values[4],perks_values[1],perks_values[6]],3,[categories_values[2]]))


#Test 4
print("Test Case 4")
print("Testing with various combinatin of categories while taking random values for other features")
print("Output with combination of category3 and perk5. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],[skills_values[2],skills_values[13]],[perks_values[5]],3,[categories_values[3]]))
print("Output with combination of category1 and skill2. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],[skills_values[2]],[perks_values[1],perks_values[6]],3,[categories_values[1]]))
print("Output with combination of category1022 and location70. Result: ",predict_values("ABC","abc@xyz",[locations_values[70]],[skills_values[2],skills_values[13]],[perks_values[5],perks_values[7]],3,[categories_values[1022]]))
print("Output with combination of category112 and perk7. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],[skills_values[2],skills_values[13]],[perks_values[7]],3,[categories_values[112]]))


#Test 5
print("Test Case 5")
print("Testing with all 72 while taking random values for other features")
for location in locations_values:
    print("Output with ",location,". Result: ",predict_values("ABC","abc@xyz",[location],[skills_values[2],skills_values[13]],[perks_values[3],perks_values[4]],3,[categories_values[2]]))


#Test 6
import random
print("Test Case 6")
print("Testing with various combination of 15 skills while taking random values for other features. \nRunning many times to ensure the results are conclusive")
def generate_random_15_skills_list():
    list = []
    for i in range(15):
        list.append(skills_values[random.randrange(267)])
    return list

print("Output with random combination of 15 skills. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],generate_random_15_skills_list(),[perks_values[3],perks_values[4]],3,[categories_values[2]]))
print("Output with random combination of 15 skills. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],generate_random_15_skills_list(),[perks_values[3],perks_values[4]],3,[categories_values[2]]))
print("Output with random combination of 15 skills. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],generate_random_15_skills_list(),[perks_values[3],perks_values[4]],3,[categories_values[2]]))
print("Output with random combination of 15 skills. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],generate_random_15_skills_list(),[perks_values[3],perks_values[4]],3,[categories_values[2]]))
print("Output with random combination of 15 skills. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],generate_random_15_skills_list(),[perks_values[3],perks_values[4]],3,[categories_values[2]]))
print("Output with random combination of 15 skills. Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],generate_random_15_skills_list(),[perks_values[3],perks_values[4]],3,[categories_values[2]]))


#Test7
print("Test Case 7")
print("Testing with all combination of perks while taking random values for other features")
print("Result: ",predict_values("ABC","abc@xyz",[locations_values[10]],[skills_values[2],skills_values[13]],[perks_values[0],perks_values[1],perks_values[2],perks_values[3],perks_values[4],perks_values[5],perks_values[6]],3,[categories_values[3]]))

def predict_location_independent(name, email, skills, perks, durations, categories):
    _ = []
    for i in range(2, len(locations_values)):
        __ = {}
        for skill in skills:
            __.update({skill: 1})

        __.update({locations_values[i]: 1})

        for perk in perks:
            if perk != 'None':
                __.update({perk: 1})

        for category in categories:
            __.update({category: 1})

        __.update({'duration': durations})

        _.append(__)
    df = pd.DataFrame(_, columns=X_columns)
    df = df.fillna(0)
    prediction = np.round(np.square(model.predict(pd.DataFrame(scaler.transform(df), columns=X_columns))), 0)
    final_dict = {}
    for i in range(0, len(prediction)):
        final_dict.update({locations_values[i + 2].split('_')[1]: prediction[i]})
    final_dict = dict(sorted(final_dict.items(), key=lambda x: x[1], reverse=True))
    keys = list(final_dict.keys())[0:10]
    values = list(final_dict.values())[0:10]
    plt.rcParams['font.size'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    ax = sns.barplot(y=keys, x=values, color='b')
    ax.bar_label(container=ax.containers[0], labels=values)
    ax.set(xticklabels=[])
    ax.set(xlabel=None)
    # ax.set(xlim=(0, 1000))
    ax.tick_params(bottom=False)
    ax.set(title='Top 10 Locations with highest stipend as per input')
    new_save_name = "result_plot_" + str(time.time()) + ".png"
    for filename in os.listdir('static/'):
        if filename.startswith("result_plot_"):
            os.remove("static/" + filename)

    plt.savefig('static/' + new_save_name, dpi=300, bbox_inches='tight')
    plt.close()
    write_log(name, email, ["Find the best one"], skills, perks, durations, categories)
    return new_save_name


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/generalised')
def generalised():
    return render_template('generalised.html')


@app.route('/customized', methods=['GET', 'POST'])
def customized():
    return render_template('customized.html', prefill=False, category=categories, category_value=categories_values,
                           len_category=len(categories), skill=skills, skill_value=skills_values, len_skill=len(skills),
                           perk=perks, perk_value=perks_values, len_perk=len(perks), location=locations,
                           location_value=locations_values, len_location=len(locations), locations_return=[],
                           skills_return=[], perks_return=[], duration_return=[], categories_return=[], predicted=0,
                           name_return=[], email_return=[], graph="False")


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        name_result = request.form.getlist("name")
        email_result = request.form.getlist('email')
        locations_result = (request.form.getlist('locations'))
        skills_result = (request.form.getlist('skills'))
        perks_result = (request.form.getlist('perks'))
        duration_result = float(request.form.getlist('duration')[0])
        categories_result = request.form.getlist('categories')
        if locations_result[0] != 'Find':
            predicted = predict_values(name_result, email_result, locations_result, skills_result, perks_result,
                                       duration_result, categories_result)
            return render_template('customized.html', prefill=True, category=categories,
                                   category_value=categories_values,
                                   len_category=len(categories), skill=skills, skill_value=skills_values,
                                   len_skill=len(skills), perk=perks, perk_value=perks_values, len_perk=len(perks),
                                   location=locations, location_value=locations_values, len_location=len(locations),
                                   locations_return=locations_result, skills_return=skills_result,
                                   perks_return=perks_result, duration_return=str(duration_result),
                                   categories_return=categories_result, name_return=name_result,
                                   email_return=email_result, predicted=predicted, graph='False')
        else:
            graph_file_name = predict_location_independent(name_result, email_result, skills_result, perks_result,
                                                           duration_result, categories_result)
            return render_template('customized.html', prefill=True, category=categories,
                                   category_value=categories_values,
                                   len_category=len(categories), skill=skills, skill_value=skills_values,
                                   len_skill=len(skills), perk=perks, perk_value=perks_values, len_perk=len(perks),
                                   location=locations, location_value=locations_values, len_location=len(locations),
                                   locations_return=locations_result, skills_return=skills_result,
                                   perks_return=perks_result, duration_return=str(duration_result),
                                   categories_return=categories_result, name_return=name_result,
                                   email_return=email_result, predicted=graph_file_name, graph='True')
