from flask import Flask, request, jsonify, render_template

import chart_studio.plotly as py
import plotly.express as px
import plotly.graph_objects as go
import json
import pandas as pd
import numpy as np
import os
import re
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from matplotlib import font_manager,rc
import matplotlib

app = Flask(__name__)
 
 
sns.set_style('darkgrid')
pd.options.display.max_rows = 150
pd.options.display.max_columns = 150
matplotlib.rcParams['axes.unicode_minus'] = False
font_path = './MALGUN.ttf'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

SC = pd.read_csv('./0514_전처리완료.csv', encoding='cp949', dtype=object)
SC['SCORE_NUM'] = SC['SCORE_NUM'].astype(float)
SC['EMP'] = SC['EMP'].astype(int)
SC['MM_all_subj'] = SC['MM_all_subj'].astype(float)
SC['MM_per_subj'] = SC['MM_per_subj'].astype(float)

def chunk_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def tmp(SC_1 = SC, dept="", scaled = True):
    main_value = "MM_per_subj" # SCORE_NUM or MM_per_subj
    result_1 = SC_1.groupby(['DEPT', 'SUBJ', 'EMP'])[main_value].agg(
        MEAN='mean',
        ).reset_index().sort_index()
    result_1['MEAN'] = result_1['MEAN'].round(4)
    # result_1
    
    result_2 = SC_1.groupby(['DEPT','SUBJ', 'EMP'])['EMP'].size().reset_index(name='CNT').sort_index()
    upper_list = []
    for name, g_df in result_2.groupby(['DEPT', 'SUBJ']):
        g_df = g_df.sort_values('EMP')
        if len(g_df) != 1:
            if (g_df.iloc[0,-1] > 2) & (g_df.iloc[1,-1] > 2):
                upper_list.append("_".join(list(name)))
    result_2 = result_2.loc[(result_2['DEPT']+"_"+result_2['SUBJ']).isin(upper_list)]
    
    result = result_1.merge(right=result_2, on=['DEPT', 'SUBJ', 'EMP'], how='inner')
    
    scaled = False if main_value == 'SCORE_NUM' else True

    rows_list = []
    for name, g_df in result.groupby(['DEPT','SUBJ'])[['CNT','EMP', 'MEAN']]:
        g_df = g_df.sort_values('EMP')
        if len(g_df) != 1:
            if (g_df.iloc[0,-3] > 2) & (g_df.iloc[1,-3] > 2):
                row = list(name) + [g_df.iloc[0,-1], g_df.iloc[1,-1]] + [g_df.iloc[0,-1] - g_df.iloc[1,-1]]
                # row = list(name) + [g_df.iloc[0,-1], g_df.iloc[1,-1]] + [(g_df.iloc[0,-1] + g_df.iloc[1,-1])/2]
                rows_list.append(row)

    rows = sorted(rows_list,key=lambda x : (-x[-1], -x[-3], x[-2]))
    rows_dict = dict([(i,r) for i, r in enumerate(rows)])

    EMP_df = pd.DataFrame(rows, columns=['DEPT', 'SUBJ', '미취업', '취업', 'GAP'])
    green_df = EMP_df[EMP_df['GAP'] == 0]

    # dept = "생명과학과"

    if dept != ['']:
        title_dept = f"{','.join(dept)}의 교과목"
        EMP_df_for_graph = EMP_df.loc[EMP_df['DEPT'].isin(dept)].reset_index(drop=True)
        # green_df_for_graph = green_df.loc[green_df['DEPT'] == dept].reset_index(drop=True)
        green_df_for_graph = EMP_df_for_graph.loc[EMP_df_for_graph['GAP'] == 0]
    else:
        title_dept = "전체 학과의 교과목"
        EMP_df_for_graph = EMP_df.reset_index(drop=True)
        green_df_for_graph = EMP_df_for_graph.loc[EMP_df['GAP'] == 0]
        
    reversal_point = EMP_df_for_graph.loc[EMP_df_for_graph['GAP'] < 0].index[0]

        
    fig = px.scatter(EMP_df_for_graph[['미취업', '취업']], 
                    trendline="lowess", 
                    text = EMP_df_for_graph['DEPT'] +"_"+ EMP_df_for_graph['SUBJ'],
                    trendline_options=dict(frac=0.25 if dept != ['']else 0.05),
                    title=f"{title_dept}\n취업VS미취업")
            
    print("1")
    fig.update_layout(
        width=1800,
        height=800,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40))      

    fig.update_traces(textfont=dict(color='rgba(0,0,0,0)'))

    print("2")
    #############################################
    # 평균차이 없음
    if len(green_df_for_graph) > 0:
        find_y_v_list = EMP_df_for_graph['미취업'].tolist() + EMP_df_for_graph['취업'].tolist()
        x0_v = green_df_for_graph.index[0]
        x1_v = green_df_for_graph.index[-1]
        y0_v, y1_v = min(find_y_v_list), max(find_y_v_list)
        
        fig.add_vrect(x0=x0_v, x1=x1_v, 
                    line_width=0, fillcolor="green", opacity=0.5,
                    annotation_text="평균차이 없음", 
                    annotation_position="bottom right",
                    annotation_x = x0_v,
                    annotation_font_size=20,
                    annotation_font_color="green",
                    annotation_font_family="Times New Roman")    
    #############################################
    # 역전점
    fig.add_vline(x=reversal_point, 
                line_width=2, 
                line_dash="dash",
                line_color="gray",
                annotation_text=f"취업 > 미취업 역전점 index : {reversal_point}", 
                annotation_position="top right",
                annotation_font_size=15,
                annotation_font_color="black")
    #############################################

    for trace in fig.data:
        if trace.mode == "lines":
            trace.y = [y_.round(4) for y_ in trace.y]
            trace.line.width = 3
            trace.hovertemplate = "추세값 : <b>%{y}</b>"
        if trace.mode == "markers+text":
            trace.marker.opacity = 0.3
            trace.marker.size = 8
            trace.hovertemplate = "<br>" + \
                                    "학과_과목명 : <b>%{text}</b><br>" + \
                                    "평균평점 : <b>%{y}</b>"

    print("3")
    # fig.data = [t for t in fig.data if t.mode == "lines"]
    fig.layout.xaxis.title.text = "교과목 인덱스"
    fig.layout.yaxis.title.text = "교과목 평균평점"
    
    graph_html = fig.to_html(full_html=False)
    print("4")
    return graph_html


@app.route('/')
def index():
    all_dept_list = list(SC.DEPT.unique())
    all_dept_list.sort()
    all_dept = chunk_list(all_dept_list, 5)
    return render_template('index.html', all_dept=all_dept)

@app.route('/search')
def search():
    query = request.args.get('query')
    print(query)
    dept = [d.strip() for d in query.split(",")]
    graph_html = tmp(SC_1 = SC, dept=dept, scaled=True)
    return jsonify({"graph_html": graph_html})

if __name__ == '__main__':
    app.run(debug=True)
