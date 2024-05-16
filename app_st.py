import streamlit as st

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
from scipy.interpolate import make_interp_spline

st.set_page_config(layout="wide")


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

def make_graphs(df,title,scaled=False):
    if scaled:
        print("스케일된 상태")
    else:
        print("스케일 안 된 상태")
        
    for_hist_df_0 = df.loc[df['EMP'] == 0, 'MEAN']
    for_hist_df_1 = df.loc[df['EMP'] == 1, 'MEAN']
    employment_map = {0: '미취업', 1: '취업'}

    display_df = df.groupby('EMP')['MEAN'].mean().reset_index()
    display_df['EMP'] = display_df['EMP'].map(employment_map)
    
    
    range_min_0= (for_hist_df_0.min()//0.25)*0.25
    range_min_1= (for_hist_df_1.min()//0.25)*0.25
    range_min = min(range_min_0, range_min_1)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    if scaled:
        range_max = 1 
        term = 0.1
    else:
        range_max = 4.5
        term = 0.2
    bins = int((range_max - range_min)//term)
        
    hist_graph = ax[0].hist((for_hist_df_0,for_hist_df_1), bins=bins, range=(range_min,range_max), label=('미취업', '취업'))
    bin_centers = (hist_graph[1][:-1] + (term/2))
    
    for i in hist_graph[0]:
        spline = make_interp_spline(bin_centers, i, k=3)  # k=3은 cubic spline을 의미함
        smooth_x = np.linspace(bin_centers.min(), bin_centers.max(), 100)
        smooth_y = [0 if ss < 0 else ss for ss in spline(smooth_x)]
        ax[0].plot(smooth_x, smooth_y)
    
    # 막대 중앙에 빈도수 표시
    for i, patches in enumerate(hist_graph[2]):  # hist_graph[2]는 각 데이터 세트의 막대(Patches) 리스트를 포함
        for rect in patches:
            height = rect.get_height()
            if height != 0: 
                ax[0].text(rect.get_x() + rect.get_width() / 2, height, f'{int(height)}',
                        ha='center', va='bottom', fontweight='bold', color='blue')

            
    ax[0].set_title(f"{title}_히스토그램")
    ax[0].set_xlabel("평균성적")
    ax[0].set_ylabel("학생 수")
    ax[0].legend()

    df['EMP_kor'] = df['EMP'].map(employment_map)
    sns.boxplot(x='EMP', y='MEAN', data=df, hue='EMP_kor', ax=ax[1], width=0.5)

    for i, row in display_df.iterrows():
        ax[1].scatter(i, row['MEAN'], color='red', s=50, zorder=5)
        
    ax[1].set_title(f"{title}_박스플롯")
    ax[1].set_xlabel("취업여부")
    ax[1].set_ylabel("평균성적")
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(['미취업', '취업'])  # x축 레이블 변경
    ax[1].legend(title='')

    return fig, ax, display_df


def tmp(tabs, SC_1 = SC, dept="", scaled = True):
    if dept:
        title_dept = f"{', '.join(dept)}의 교과목"
        SC_1 = SC_1.loc[SC_1['DEPT'].isin(dept)].reset_index(drop=True)
    else:
        title_dept = "전체 학과의 교과목"
        SC_1 = SC_1.reset_index(drop=True)
        
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
    fig1, ax1, display_df = make_graphs(result, "과목", scaled)
    
    rows_list = []
    for name, g_df in result.groupby(['DEPT','SUBJ'])[['CNT','EMP', 'MEAN']]:
        g_df = g_df.sort_values('EMP')
        if len(g_df) != 1:
            if (g_df.iloc[0,-3] > 2) & (g_df.iloc[1,-3] > 2):
                row = list(name) + [g_df.iloc[0,-1], g_df.iloc[1,-1]] + [g_df.iloc[0,-1] - g_df.iloc[1,-1]]
                # row = list(name) + [g_df.iloc[0,-1], g_df.iloc[1,-1]] + [(g_df.iloc[0,-1] + g_df.iloc[1,-1])/2]
                rows_list.append(row)

    rows = sorted(rows_list,key=lambda x : (-x[-1], -x[-3], x[-2]))

    EMP_df = pd.DataFrame(rows, columns=['DEPT', 'SUBJ', '미취업', '취업', 'GAP'])

    EMP_df_for_graph = EMP_df.copy()
    green_df_for_graph = EMP_df_for_graph.loc[EMP_df_for_graph['GAP'] == 0]
    reversal_point = EMP_df_for_graph.loc[EMP_df_for_graph['GAP'] < 0].index[0]
    
    EMP_df_for_graph['GAP_2'] = EMP_df_for_graph['GAP']/2
    EMP_df_for_graph['미취업_2'] = 0.5 + EMP_df_for_graph['GAP_2']
    EMP_df_for_graph['취업_2'] = 0.5 - EMP_df_for_graph['GAP_2']

    dept_join_subj = EMP_df_for_graph['DEPT'] +"_"+ EMP_df_for_graph['SUBJ']
    cnt_dept = f"{title_dept} 개수 : {dept_join_subj.nunique()}개"
    
    with tabs[0]:
        st.dataframe(display_df)
        st.pyplot(fig1)
        st.subheader(cnt_dept)
    
    cols_list = [['미취업', '취업'], ['미취업_2', '취업_2']]
    for cdx, cols in enumerate(cols_list):
        fig = px.scatter(EMP_df_for_graph[cols], 
                        trendline="lowess", 
                        text = dept_join_subj,
                        trendline_options=dict(frac=0.25 if dept else 0.05),
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
            find_y_v_list = EMP_df_for_graph[cols[0]].tolist() + EMP_df_for_graph[cols[1]].tolist()
            x0_v = green_df_for_graph.index[0]
            x1_v = green_df_for_graph.index[-1]
            y0_v, y1_v = min(find_y_v_list), max(find_y_v_list)
            
            fig.add_vrect(x0=x0_v, x1=x1_v, 
                        line_width=2, fillcolor="green", opacity=0.5, line_color="green",
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
                                        "학과_과목명 : <b>%{text}</b><br>"
                if cdx == 0:
                    trace.hovertemplate += "평균평점 : <b>%{y}</b>"
                else:
                    trace.hovertemplate += "정규화점수 : <b>%{y}</b>"
                                        
            if trace.legendgroup in ['취업', '취업_2']:
                trace.marker.color = "#EF553B"
            else:
                trace.marker.color = "#636efa"

        print("3")
        # fig.data = [t for t in fig.data if t.mode == "lines"]
        fig.layout.xaxis.title.text = "교과목 인덱스"
        fig.layout.yaxis.title.text = "교과목 평균평점" if cdx == 0 else "교과목 정규화점수"
        
        # graph_html = fig.to_html(full_html=False)
        print("4")
        with tabs[cdx+1]:
            st.plotly_chart(fig)
            st.subheader(cnt_dept)
        # return graph_html

all_dept_list = list(SC.DEPT.unique())
all_dept_list.sort()

st.header("학과 선택")
dept = st.multiselect(" ", all_dept_list, label_visibility='collapsed')
start = st.button("분석 시작")  
st.write("* 전체 학과로 검색하려면 공란으로 검색하세요.")
st.write("* 소요시간 : 5초 내외")

st.divider()

if start:
    tabs = st.tabs(["**기본 그래프**", "**평균평점 세부 그래프**", "**정규화점수 세부 그래프**"])
    tmp(SC_1 = SC, dept=dept, scaled=True, tabs = tabs)



