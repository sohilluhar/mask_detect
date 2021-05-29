import base64, urllib
import io, json

from django.core.files.storage import FileSystemStorage
from django.http import HttpResponseRedirect
from django.shortcuts import render
from matplotlib import image


def login(request):
    return render(request, 'login.html', {})


def verify(request):
    email = request.POST.get('email')
    password = request.POST.get('password')
    if email == "admin@gmail.com" and password == "admin":
        return HttpResponseRedirect('home')
    else:
        return render(request, 'redirect.html',
                      {"swicon": "error", "swtitle": "Error", "swmsg": "Invalid Password or username",
                       "path": "login"})


def home(request):
    return render(request, 'home.html', {'title': 'Hello'})


def prediction(req):
    return render(req, 'prediction.html', {})


def chart(req):
    return render(req, 'graph.html', {})


def field_distributions(req):
    import numpy as np  # linear algebra
    import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
    import os
    from glob import glob

    import matplotlib.pyplot as plt
    import seaborn as sns

    ##Import any other packages you may need here
    import matplotlib.image as image

    all_xray_df = pd.read_csv('Data_Entry_2017.csv')
    sample_df = pd.read_csv('sample_labels.csv')

    all_xray_df['Patient Age'] = all_xray_df.apply(lambda x: 101 if x['Patient Age'] > 100 else x['Patient Age'],
                                                   axis=1)
    over100 = all_xray_df[all_xray_df['Patient Age'] > 100]
    print("\nAge\n" + str(len(over100)))
    # h = all_xray_df['Patient Age'].hist(bins=100)

    plt.style.use('fivethirtyeight')

    fig1, ax = plt.subplots()
    all_xray_df['Patient Age'].hist(bins=100, ax=ax)
    # plt.plot(all_xray_df['Patient Age'].hist(bins=100), label='AGE')
    plt.title('AGE')
    plt.xlabel('Age')
    plt.ylabel('No of Patient')
    plt.legend(loc='upper left')
    # fig1 = plt.gcf()
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    buf1.seek(0)
    str1 = base64.b64encode(buf1.read())
    uri1 = urllib.parse.quote(str1)
    # uri1 = ""

    # GENDER
    plt.clf()
    gender_m = all_xray_df[all_xray_df['Patient Gender'] == 'M']
    gender_f = all_xray_df[all_xray_df['Patient Gender'] == 'F']
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(12.5, 7))
    plt.bar(['Male', 'Female'], [len(gender_m), len(gender_f)])

    fig2 = plt.gcf()
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png')
    buf2.seek(0)
    str2 = base64.b64encode(buf2.read())
    uri2 = urllib.parse.quote(str2)

    # View Position
    all_xray_df['View Position'].unique()
    pos_pa = all_xray_df[all_xray_df['View Position'] == 'PA']
    pos_ap = all_xray_df[all_xray_df['View Position'] == 'AP']
    plt.clf()
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(12.5, 7))
    plt.bar(['PA', 'AP'], [len(pos_pa), len(pos_ap)])

    fig2 = plt.gcf()
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png')
    buf2.seek(0)
    str3 = base64.b64encode(buf2.read())
    uri3 = urllib.parse.quote(str3)

    # Follow Up
    plt.clf()
    fig1, ax = plt.subplots()
    # df.hist('ColumnName', ax=ax)
    # all_xray_df['Patient Age'].hist(bins=100, ax=ax)
    all_xray_df['Follow-up #'].hist(bins=200, ax=ax)
    # plt.plot(all_xray_df['Patient Age'].hist(bins=100), label='AGE')

    # fig1 = plt.gcf()
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    buf1.seek(0)
    str1 = base64.b64encode(buf1.read())
    uri4 = urllib.parse.quote(str1)
    all_xray_df['Follow-up #'].unique()
    follow_up_0 = all_xray_df[all_xray_df['Follow-up #'] == 0]
    print(f'Follow-up number 0 (first visit): {len(follow_up_0)} ({100.0 * len(follow_up_0) / len(all_xray_df) :.2f}%)')

    follow_up_1 = all_xray_df[all_xray_df['Follow-up #'] == 1]
    print(
        f'Follow-up number 1 (second visit): {len(follow_up_1)} ({100.0 * len(follow_up_1) / len(all_xray_df) :.2f}%)')

    follow_up_2 = all_xray_df[all_xray_df['Follow-up #'] == 2]
    print(
        f'Follow-up number 2 (second visit): {len(follow_up_2)} ({100.0 * len(follow_up_2) / len(all_xray_df) :.2f}%)')

    follow_up_3 = all_xray_df[all_xray_df['Follow-up #'] == 3]
    print(
        f'Follow-up number 3 (second visit): {len(follow_up_3)} ({100.0 * len(follow_up_3) / len(all_xray_df) :.2f}%)')

    follow_up_4 = all_xray_df[all_xray_df['Follow-up #'] == 4]
    print(
        f'Follow-up number 4 (second visit): {len(follow_up_4)} ({100.0 * len(follow_up_4) / len(all_xray_df) :.2f}%)')

    per_follow_up_0 = f'{100.0 * len(follow_up_0) / len(all_xray_df): .2f}%'
    per_follow_up_1 = f'{100.0 * len(follow_up_1) / len(all_xray_df): .2f}%'
    per_follow_up_2 = f'{100.0 * len(follow_up_2) / len(all_xray_df): .2f}%'
    per_follow_up_3 = f'{100.0 * len(follow_up_3) / len(all_xray_df): .2f}%'
    per_follow_up_4 = f'{100.0 * len(follow_up_4) / len(all_xray_df): .2f}%'

    # Finding Labels

    print("********Finding Labels************")
    all_xray_df['Finding Labels'].unique()
    all_xray_df['Finding Labels'].nunique()

    findings = set()
    for f in all_xray_df['Finding Labels'].unique():
        findings.update(f.split('|'))
    print_str1 = f'Total number of single diagnoses: {len(findings)}'
    no_finding = all_xray_df[all_xray_df['Finding Labels'] == 'No Finding']
    print_str2 = f'No finding: {len(no_finding)} ({100.0 * len(no_finding) / len(all_xray_df) :.2f}%)'

    # Patient ID
    print("********Patient ID************")
    unique_patients_num = all_xray_df['Patient ID'].nunique()
    patient_print0 = f'Total unique patients: {unique_patients_num}, average number records per patient: {len(all_xray_df) / unique_patients_num :.2f}'
    records_per_patient = []
    for pid in all_xray_df['Patient ID'].unique():
        records_per_patient.append(len(all_xray_df[all_xray_df['Patient ID'] == pid]))

    plt.clf()
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(12.5, 7))

    plt.hist(records_per_patient, bins=max(records_per_patient))

    fig1 = plt.gcf()
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    buf1.seek(0)
    str1 = base64.b64encode(buf1.read())
    uri5 = urllib.parse.quote(str1)

    def num_patients_with_records(num):
        return (np.array(records_per_patient) == num * np.ones(len(records_per_patient))).sum()

    patient_print1 = ''
    for i in range(1, 11):
        patient_print1 = patient_print1 + f'Number of patients with {i} records in the dataset: {num_patients_with_records(i)} ({100.0 * num_patients_with_records(i) / unique_patients_num :.2f}%)' + '\n'

    # Image width & height, pixel spacing
    print("*****************Image width & height, pixel spacing*******")
    desc = all_xray_df.describe()

    def show_min_max_values(col, name):
        return f'{name} range: [{desc[col]["min"]}, {desc[col]["max"]}]'

    image1_str = show_min_max_values('OriginalImage[Width', 'Image Width')
    image2_str = show_min_max_values('Height]', 'Image Height')
    image3_str = show_min_max_values('OriginalImagePixelSpacing[x', 'Pixel Spacing over X')
    image4_str = show_min_max_values('y]', 'Pixel Spacing over Y')
    image_str_final1 = image1_str + '\n' + image2_str + '\n' + image3_str + '\n' + image4_str + '\n'

    plt.clf()
    fig1, ax = plt.subplots()
    all_xray_df['OriginalImage[Width'].hist(bins=100, ax=ax)
    plt.title('Width')
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    buf1.seek(0)
    str1 = base64.b64encode(buf1.read())
    uri6 = urllib.parse.quote(str1)

    plt.clf()
    fig1, ax = plt.subplots()
    all_xray_df['Height]'].hist(bins=100, ax=ax)
    plt.title('Height')
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    buf1.seek(0)
    str1 = base64.b64encode(buf1.read())
    uri7 = urllib.parse.quote(str1)

    width2500 = all_xray_df[(all_xray_df['OriginalImage[Width'] == 2500)]
    width2500_str = (f'Images with width 2500: {len(width2500)} ({100.0 * len(width2500) / len(all_xray_df) :.2f}%)')
    height2048 = all_xray_df[(all_xray_df['Height]'] == 2048)]
    height2048_str = (
        f'Images with height 2048: {len(height2048)} ({100.0 * len(height2048) / len(all_xray_df) :.2f}%)')
    size2500x2048 = all_xray_df[(all_xray_df['OriginalImage[Width'] == 2500) & (all_xray_df['Height]'] == 2048)]
    size2500x2048_str = (
        f'Images 2500x2048: {len(size2500x2048)} ({100.0 * len(size2500x2048) / len(all_xray_df) :.2f}%)')
    image_str_final2 = width2500_str + '\n' + height2048_str + '\n' + size2500x2048_str

    return render(req, 'field_distributions.html',
                  {'age_plot': uri1, 'gender': uri2, 'view_position': uri3, 'followup': uri4,
                   'per_follow_up_0': per_follow_up_0, 'follow_up_0': len(follow_up_0),
                   'per_follow_up_1': per_follow_up_1, 'follow_up_1': len(follow_up_1),
                   'per_follow_up_2': per_follow_up_2, 'follow_up_2': len(follow_up_2),
                   'per_follow_up_3': per_follow_up_3, 'follow_up_3': len(follow_up_3),
                   'per_follow_up_4': per_follow_up_4, 'follow_up_4': len(follow_up_4),
                   'print_str1': print_str1, 'print_str2': print_str2, 'findings': findings,
                   'patient_id_graph': uri5, 'patient_print0': patient_print0, 'patient_print1': patient_print1,
                   'image_str_final1': image_str_final1, 'img_graph1': uri6, 'img_graph2': uri7,
                   'image_str_final2': image_str_final2
                   })


def pneumonia_cases(request):
    import numpy as np  # linear algebra
    import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
    import os
    from glob import glob
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    all_xray_df = pd.read_csv('Data_Entry_2017.csv')
    all_xray_df['Patient Age'] = all_xray_df.apply(lambda x: 101 if x['Patient Age'] > 100 else x['Patient Age'],
                                                   axis=1)
    over100 = all_xray_df[all_xray_df['Patient Age'] > 100]
    # Map findings per single finding
    print("****************Map findings per single finding***************")
    # sample_df = pd.read_csv('sample_labels.csv')
    all_xray_df['Finding Labels'].unique()
    all_xray_df['Finding Labels'].nunique()

    findings = set()
    for f in all_xray_df['Finding Labels'].unique():
        findings.update(f.split('|'))

    for finding in findings:
        all_xray_df[finding] = all_xray_df['Finding Labels'].map(lambda x: 1.0 if finding in x else 0)
    pneumonia = all_xray_df[all_xray_df['Pneumonia'] == 1]
    all_findings = all_xray_df[all_xray_df["No Finding"] == 0]
    print(f'All findings: {len(all_findings)}')
    single_finding_str1 = (f'All findings: {len(all_findings)}')

    print(f'Pneumonia images: {len(pneumonia)} ({100.0 * len(pneumonia) / len(all_xray_df) :.2f}% of all)')
    print(f'Pneumonia images: {len(pneumonia)} ({100.0 * len(pneumonia) / len(all_findings) :.2f}% of findings)')
    single_finding_str2 = (
        f'Pneumonia images: {len(pneumonia)} ({100.0 * len(pneumonia) / len(all_xray_df) :.2f}% of all)')
    single_finding_str3 = (
        f'Pneumonia images: {len(pneumonia)} ({100.0 * len(pneumonia) / len(all_findings) :.2f}% of findings)')

    no_pneumonia = all_xray_df[all_xray_df["Pneumonia"] == 0]
    print(f'No pneumonia: {len(no_pneumonia)}')
    single_finding_str4 = (f'No pneumonia: {len(no_pneumonia)}')
    no_pneumonia_findings = all_xray_df[(all_xray_df["Pneumonia"] == 0) & (all_xray_df["No Finding"] == 0)]
    print(f'No pneumonia among findings: {len(no_pneumonia_findings)}')
    single_finding_str5 = (f'No pneumonia among findings: {len(no_pneumonia_findings)}')
    final_str = single_finding_str1 + '\n\n' + single_finding_str2 + '\n' + single_finding_str3 + '\n\n' + single_finding_str4 + '\n' + single_finding_str5

    # Feature distributions among pneumonia records
    print("***************Feature distributions among pneumonia records******************")

    # pneumonia['Patient Age'].hist(bins=100)
    plt.clf()
    fig1, ax = plt.subplots()
    pneumonia['Patient Age'].hist(bins=100, ax=ax)
    plt.title('Age')
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    buf1.seek(0)
    str1 = base64.b64encode(buf1.read())
    uri1 = urllib.parse.quote(str1)

    # pneumonia['Patient Gender'].hist(bins=2)
    plt.clf()
    fig1, ax = plt.subplots()
    pneumonia['Patient Gender'].hist(bins=100, ax=ax)
    plt.title('Gender')
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    buf1.seek(0)
    str1 = base64.b64encode(buf1.read())
    uri2 = urllib.parse.quote(str1)

    # pneumonia['View Position'].hist(bins=2)

    plt.clf()
    fig1, ax = plt.subplots()
    pneumonia['View Position'].hist(bins=100, ax=ax)
    plt.title('Position')
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    buf1.seek(0)
    str1 = base64.b64encode(buf1.read())
    uri3 = urllib.parse.quote(str1)

    # Patient ID
    pneumonia_unique_patients_num = pneumonia['Patient ID'].nunique()
    print(
        f'Total pneumonia patients: {pneumonia_unique_patients_num}, average number records per patient: {len(pneumonia) / pneumonia_unique_patients_num :.2f}')
    patient_str1 = (
        f'Total pneumonia patients: {pneumonia_unique_patients_num}, average number records per patient: {len(pneumonia) / pneumonia_unique_patients_num :.2f}')

    pneumonia_records_per_patient = []
    for pid in pneumonia['Patient ID'].unique():
        pneumonia_records_per_patient.append(len(pneumonia[pneumonia['Patient ID'] == pid]))

    plt.clf()
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(12.5, 7))

    plt.hist(pneumonia_records_per_patient, bins=max(pneumonia_records_per_patient))

    fig1 = plt.gcf()
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    buf1.seek(0)
    str1 = base64.b64encode(buf1.read())
    uri4 = urllib.parse.quote(str1)

    # pneumonia['Follow-up #'].hist(bins=200)

    plt.clf()
    fig1, ax = plt.subplots()
    pneumonia['Follow-up #'].hist(bins=200, ax=ax)

    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    buf1.seek(0)
    str1 = base64.b64encode(buf1.read())
    uri5 = urllib.parse.quote(str1)

    # Pneumonia & comorbid findings

    plt.clf()
    fig1, ax = plt.subplots()
    ax = all_xray_df[findings].sum().sort_values(ascending=False).plot(kind='bar')
    ax.set(ylabel='Number of Images with Label')
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    buf1.seek(0)
    str1 = base64.b64encode(buf1.read())
    uri6 = urllib.parse.quote(str1)

    data1 = (all_xray_df[findings].sum() / len(all_xray_df)).sort_values(ascending=False)

    plt.clf()
    plt.figure(figsize=(16, 6))
    fig1, ax = plt.subplots()
    ax = all_xray_df[all_xray_df['Pneumonia'] == 1]['Finding Labels'].value_counts()[0:30].plot(kind='bar')
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    buf1.seek(0)
    str1 = base64.b64encode(buf1.read())
    uri7 = urllib.parse.quote(str1)

    data2 = all_xray_df[all_xray_df['Pneumonia'] == 1]['Finding Labels'].value_counts()[0:30]

    # ### Frequency of comorbid conditions with pneumonia (per each condition)

    plt.clf()
    plt.figure(figsize=(16, 6))
    fig1, ax = plt.subplots()
    ax = pneumonia[findings].sum().sort_values(ascending=False).plot(kind='bar')
    ax.set(ylabel='Number of Images with Label')
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    buf1.seek(0)
    str1 = base64.b64encode(buf1.read())
    uri8 = urllib.parse.quote(str1)
    # Number of deseases per patient
    plt.clf()
    fig1, ax = plt.subplots()
    pneumonia[findings].sum(axis=1).hist(bins=10, ax=ax)
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    buf1.seek(0)
    str1 = base64.b64encode(buf1.read())
    uri9 = urllib.parse.quote(str1)

    plt.clf()
    fig1, ax = plt.subplots()
    all_findings[findings].sum(axis=1).hist(bins=10, ax=ax)
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    buf1.seek(0)
    str1 = base64.b64encode(buf1.read())
    uri10 = urllib.parse.quote(str1)

    return render(request, 'pneumonia-cases.html',
                  {'final_str': final_str,
                   'age': uri1,
                   'gender': uri2,
                   'position': uri3,
                   'patient_str': patient_str1,
                   'patient_graph': uri4,
                   'follow_up': uri5,
                   'comorbid': uri6,
                   'data1': data1,
                   'comorbid2': uri7,
                   'data2': data2,
                   'comorbidconditions': uri8,
                   'deseasesperpatient': uri9,
                   'deseasesperpatient1': uri10,

                   }

                  )


def poixel_analysis(request):
    # import numpy as np  # linear algebra
    # import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
    # import os
    # from glob import glob
    # import matplotlib
    # matplotlib.use('Agg')
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    #
    # all_xray_df = pd.read_csv('Data_Entry_2017.csv')
    # all_xray_df['Patient Age'] = all_xray_df.apply(lambda x: 101 if x['Patient Age'] > 100 else x['Patient Age'],
    #                                                axis=1)
    # over100 = all_xray_df[all_xray_df['Patient Age'] > 100]
    # # Map findings per single finding
    # print("****************Map findings per single finding***************")
    # sample_df = pd.read_csv('sample_labels.csv')
    #
    # findings = set()
    # for f in all_xray_df['Finding Labels'].unique():
    #     findings.update(f.split('|'))
    # for finding in findings:
    #     sample_df[finding] = sample_df['Finding Labels'].map(lambda x: 1.0 if finding in x else 0)
    #
    # def get_image_path(row):
    #     fpath = None
    #     f = row['Image Index']
    #     for d in range(12):
    #         dname = 'images_' + str(d).zfill(3)
    #         fname = '/data/' + dname + '/images/' + f
    #         if os.path.isfile(fname):
    #             fpath = fname
    #             break
    #     return fpath
    #
    # sample_df['image_path'] = sample_df.apply(get_image_path, axis=1)
    # pneumo_samples = sample_df[sample_df['Pneumonia'] == 1]
    # pneumonia_example_1 = pneumo_samples.iloc[0]
    #
    # def show_image_distr(img_data):
    #
    #     plt.clf()
    #     f = plt.figure()
    #     f.set_figwidth(10)
    #
    #     s1 = f.add_subplot(1, 2, 1)
    #     s1.set_title('Image')
    #     plt.imshow(img_data, cmap='gray')
    #
    #     s2 = f.add_subplot(1, 2, 2)
    #     s2.set_title('Intensity Distribution')
    #     plt.hist(img_data.ravel(), bins=256)
    #     # plt.show()
    #
    #     fig1 = plt.gcf()
    #     buf1 = io.BytesIO()
    #     fig1.savefig(buf1, format='png')
    #     buf1.seek(0)
    #     str1 = base64.b64encode(buf1.read())
    #     uri = urllib.parse.quote(str1)
    #     return uri
    #
    # ex1_data = image.imread('/data/images_002/images/' + pneumonia_example_1['Image Index'])
    # show_image_distr(ex1_data)

    return render(request, 'pixel.html',
                  {}
                  )


def senti(req):
    return render(req, 'home.html', {'title': 'Hello'})


def news(req):
    return render(req, 'news.html', {})


def test(req):
    import numpy as np  # linear algebra
    import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
    import os
    from glob import glob

    import matplotlib.pyplot as plt
    import seaborn as sns

    ##Import any other packages you may need here
    import matplotlib.image as image

    all_xray_df = pd.read_csv('Data_Entry_2017.csv')
    # sample_df = pd.read_csv('sample_labels.csv')

    all_xray_df['Patient Age'] = all_xray_df.apply(lambda x: 101 if x['Patient Age'] > 100 else x['Patient Age'],
                                                   axis=1)
    over100 = all_xray_df[all_xray_df['Patient Age'] > 100]
    print("\nAge\n" + str(len(over100)))

    fig, ax = plt.subplots()
    # df.hist('ColumnName', ax=ax)
    all_xray_df['Patient Age'].hist(bins=100, ax=ax)
    fig.savefig('example.png')

    # plt.style.use('fivethirtyeight')
    #
    # plt.figure(figsize=(12.5, 7))
    # fig, ax = plt.subplots()
    # # df.hist('ColumnName', ax=ax)
    # all_xray_df['Patient Age'].hist(bins=100, ax=ax)
    # # plt.plot(all_xray_df['Patient Age'].hist(bins=100), label='AGE')
    # plt.title('AGE')
    # plt.xlabel('Age')
    # plt.ylabel('No of Patient')
    # plt.legend(loc='upper left')
    # fig1 = plt.gcf()
    # buf1 = io.BytesIO()
    # fig1.savefig(buf1, format='png')
    # buf1.seek(0)
    # str1 = base64.b64encode(buf1.read())
    # uri1 = urllib.parse.quote(str1)

    return render(req, 'field_distributions.html', {'age_plot': ''})


def map(request):
    import requests

    # type = request.GET['type']
    # print("Type is " + type)

    URL = "https://discover.search.hereapi.com/v1/discover"
    latitude = 23.2631191
    longitude = 69.6599372

    latitude = 19.0940581
    longitude = 72.8966304

    # Acquire from developer.here.com
    # api_key = "7lEw8DatD-WuPNIZPWSpCilQkYk3Stglhq3l60gJav0"
    api_key = "OzY0jbh2SnqI5LeBhsvblvlLpa4xKet3XPTEngeyVAs"
    query = 'hospital'
    limit = 5

    PARAMS = {
        'apikey': api_key,
        'q': query,
        'limit': limit,
        'at': '{},{}'.format(latitude, longitude)
    }

    # sending get request and saving the response as response object
    r = requests.get(url=URL, params=PARAMS)
    data = r.json()
    print(data)
    # print(data['items'][0]['categories'][0]['name'])

    hospitalOne = ''
    hospitalOne_address = ''
    hospitalOne_contect = ''
    hospitalOne_latitude = ''
    hospitalOne_longitude = ''
    hospitalOne_Docter_name = ''

    hospitalTwo = ''
    hospitalTwo_address = ''
    hospitalTwo_contect = ''
    hospitalTwo_latitude = ''
    hospitalTwo_longitude = ''
    hospitalTwo_Docter_name = ''

    try:
        hospitalOne = data['items'][0]['title']
        hospitalOne_address = data['items'][0]['address']['label']
        hospitalOne_latitude = data['items'][0]['position']['lat']
        hospitalOne_longitude = data['items'][0]['position']['lng']
        hospitalOne_contect = data['items'][0]['contacts'][0]
        hospitalOne_Docter_name = data['items'][0]['categories'][0]['name']

        hospitalTwo = data['items'][1]['title']
        hospitalTwo_address = data['items'][1]['address']['label']
        hospitalTwo_latitude = data['items'][1]['position']['lat']
        hospitalTwo_longitude = data['items'][1]['position']['lng']
        hospitalTwo_contect = data['items'][1]['contacts'][0]
        hospitalTwo_Docter_name = data['items'][0]['categories'][0]['name']
    except:
        print("error")

    print(hospitalOne, hospitalOne_address, hospitalOne_contect)
    print(hospitalTwo, hospitalTwo_address, hospitalTwo_contect)
    return render(request, 'map1.html', {
        'lat': latitude, 'long': longitude,
        'hospitalOne': hospitalOne, 'hospitalOne_address': hospitalOne_address,
        'hospitalOne_contect': hospitalOne_contect, 'hospitalOne_Docter_name': hospitalOne_Docter_name,
        'hospitalOne_latitude': hospitalOne_latitude, 'hospitalOne_longitude': hospitalOne_longitude,
        'hospitalTwo': hospitalTwo, 'hospitalTwo_address': hospitalTwo_address,
        'hospitalTwo_contect': hospitalTwo_contect, 'hospitalTwo_Docter_name': hospitalTwo_Docter_name,
        'hospitalTwo_latitude': hospitalTwo_latitude, 'hospitalTwo_longitude': hospitalTwo_longitude
    })


def upload(request):
    # print("in upload")
    if request.method == 'POST' and request.FILES['scan']:
        myfile = request.FILES['scan']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)

        # print(uploaded_file_url)
        imgpath = 'media/' + filename
        # Prediction CODE here
        # *********************
        # from keras.models import load_model
        # from tensorflow.keras.preprocessing import image
        # import numpy as np
        # import pandas as pd
        # import random
        # import cv2
        # import matplotlib.pyplot as plt
        # img_dims = 150
        # loaded_model = load_model('xray_model2.h5')
        # img = plt.imread("chest-example2.png")
        # img = cv2.resize(img, (img_dims, img_dims))
        # img = np.dstack([img, img, img])
        # img = img.astype('float32') / 255
        #
        #
        # img = np.array([img])
        # # new_image.shape
        # preds = loaded_model.predict(img)

        preds = 0.8
        preds = preds * 100
        print(preds)
        from shutil import copyfile
        copyfile('media/' + filename, 'static/media/' + filename)

        return render(request, 'prediction.html', {'acc': preds, 'imgpathhtml': 'media/' + filename})
    else:
        return render(request, 'prediction.html', {})
