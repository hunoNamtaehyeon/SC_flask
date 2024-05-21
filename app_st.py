import streamlit as st

import chart_studio.plotly as py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import pandas as pd
import numpy as np
import os
import re
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager,rc
import matplotlib
from scipy.interpolate import make_interp_spline

st.set_page_config(layout="wide")

sns.set_style('darkgrid')
pd.options.display.max_rows = 150
pd.options.display.max_columns = 150

st.header('학교 선택')
schl = st.radio(' ', ['순천향대학교', '청주대학교'], label_visibility = 'collapsed', horizontal = True)

if 'SC' not in st.session_state:
    SC = pd.read_csv('./0514_전처리완료.csv', encoding='cp949', dtype=object)
    SC['SCORE_NUM'] = SC['SCORE_NUM'].astype(float)
    SC['EMP'] = SC['EMP'].astype(int)
    SC['MM_all_subj'] = SC['MM_all_subj'].astype(float)
    SC['MM_per_subj'] = SC['MM_per_subj'].astype(float)
    st.session_state['SC'] = SC
    
if 'CJ' not in st.session_state:
    # CJ = pd.read_csv('./0520_청주_전처리완료.csv', encoding='cp949', dtype=object)
    CJ = pd.read_csv('./0521_청주_신입학만_전처리완료.csv', encoding='cp949', dtype=object)
    CJ['SCORE_NUM'] = CJ['SCORE_NUM'].astype(float)
    CJ['EMP'] = CJ['EMP'].astype(int)
    CJ['MM_all_subj'] = CJ['MM_all_subj'].astype(float)
    CJ['MM_per_subj'] = CJ['MM_per_subj'].astype(float)
    st.session_state['CJ'] = CJ

if schl == '순천향대학교':
    DF = st.session_state['SC']
else:
    DF = st.session_state['CJ']




def make_graphs(df,title,select_term,tabs,scaled=False):
    if scaled:
        print("스케일된 상태")
    else:
        print("스케일 안 된 상태")
        
    for_hist_df_0 = df.loc[df['EMP'] == 0, 'MEAN']
    for_hist_df_1 = df.loc[df['EMP'] == 1, 'MEAN']
    employment_map = {0: '미취업', 1: '취업'}

    display_df = df.groupby('EMP')['MEAN'].mean().reset_index()
    display_df['EMP'] = display_df['EMP'].map(employment_map)
    
    # display(display_df)
    
    range_min_0= (for_hist_df_0.min()//0.25)*0.25
    range_min_1= (for_hist_df_1.min()//0.25)*0.25
    range_min = min(range_min_0, range_min_1)

    if scaled:
        range_max = 1 
        term = 0.1
    else:
        range_max = 4.5
        term = 0.2
    bins = int((range_max - range_min)//term)
            
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"{title}_히스토그램", f"{title}_박스플롯"), specs=[[{"type": "histogram"}, {"type": "box"}]])
    fig.add_trace(go.Histogram(x=for_hist_df_0, name='미취업', nbinsx=select_term, texttemplate="%{y}", textposition="outside", marker_color='blue'), row=1, col=1)
    fig.add_trace(go.Histogram(x=for_hist_df_1, name='취업', nbinsx=select_term, texttemplate="%{y}", textposition="outside", marker_color='red'), row=1, col=1)

    fig.add_trace(go.Violin(y=for_hist_df_0, name="미취업", marker_color='blue', meanline_visible=True, box_visible=True, showlegend=False), row=1, col=2)
    fig.add_trace(go.Violin(y=for_hist_df_1, name="취업", marker_color='red', meanline_visible=True, box_visible=True, showlegend=False), row=1, col=2)

    fig.update_layout(
        width=1800,
        height=800,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40))  

    
    
    fig.layout.xaxis.title.text = "교과목 정규화점수" if scaled else "교과목 평균평점" 
    fig.layout.yaxis.title.text = "인원 수"
    fig.layout.xaxis2.title.text = "취업여부"
    fig.layout.yaxis2.title.text = "교과목 정규화점수" if scaled else "교과목 평균평점" 
    with tabs[0]:
        st.dataframe(display_df)
    st.session_state['figs'][0] = fig  


def make_pie_graph(EMP_df_for_graph):
    pie_hover_list = []

    EMP_df_for_graph['GAP_2'] = EMP_df_for_graph['GAP']/2
    EMP_df_for_graph['미취업_2'] = 0.5 + EMP_df_for_graph['GAP_2']
    EMP_df_for_graph['취업_2'] = 0.5 - EMP_df_for_graph['GAP_2']
    
    over_df_1 = EMP_df_for_graph[(EMP_df_for_graph.GAP < 0)]
    same_df   = EMP_df_for_graph[(EMP_df_for_graph.GAP == 0)]
    over_df_0 = EMP_df_for_graph[(EMP_df_for_graph.GAP > 0)]

    div_list = list(EMP_df_for_graph.DIV.unique())

    for chk_df in [over_df_1, same_df, over_df_0]:
        piece_dict = dict({div : 0 for div in div_list})
        for u_d in div_list:
            div_cnt = len(chk_df.loc[chk_df['DIV'] == u_d])
            piece_dict[u_d] = div_cnt
        pie_hover_list.append(piece_dict)

    over_df_1 = EMP_df_for_graph[(EMP_df_for_graph.GAP < 0)]
    same_df   = EMP_df_for_graph[(EMP_df_for_graph.GAP == 0)]
    over_df_0 = EMP_df_for_graph[(EMP_df_for_graph.GAP > 0)]

    div_list = list(EMP_df_for_graph.DIV.unique())

    # 값과 이름 리스트를 생성합니다.
    values = [len(over_df_1), len(same_df), len(over_df_0)]
    names = ['취업 > 미취업','취업=미취업','취업 < 미취업']
    colors = ['red', 'green', 'blue']

    # 값이 0이 아닌 데이터만 필터링합니다.
    filtered_values = [value for value in values if value != 0]
    filtered_names = [names[i] for i in range(len(values)) if values[i] != 0]
    filtered_colors = [colors[i] for i in range(len(values)) if values[i] != 0]
    filtered_extra_info = [pie_hover_list[i] for i in range(len(values)) if values[i] != 0]

    # 필터링된 데이터를 사용하여 파이 차트를 생성합니다.
    pie_fig = px.pie(values=filtered_values, 
                    names=filtered_names,
                    title="점수 우위 과목 수")

    # customdata에 추가 정보를 저장합니다.
    customdata = [[e_v for e_v in extra.values()] for extra in filtered_extra_info]

    hover_text = '전체=%{value}<br>'

    for ddx, div in enumerate(div_list):
        div_text = f'{div_list[ddx]}'+'=%{customdata[0]['+str(ddx)+']}'
        hover_text += div_text
        if ddx != len(div_list)-1:
            hover_text += '<br>'

    hover_text += '<extra></extra>'

    # 트레이스를 업데이트합니다.
    pie_fig.update_traces(
        textfont_size=15,
        marker_colors=filtered_colors,
        marker_line_color="black",
        textposition='outside',
        textinfo='label+percent+value',
        customdata=customdata,
        hovertemplate=hover_text
    )

    # 레이아웃을 업데이트합니다.
    pie_fig.update_layout(font=dict(size=15))

    # # 그래프를 출력합니다.
    # pie_fig.show()
    st.session_state['figs'][1] = pie_fig
    
    return EMP_df_for_graph

def tmp(tabs, select_term, DF_1 = DF, dept="", scaled = True):
    if dept:
        title_dept = f"{', '.join(dept)}의 교과목"
        DF_1 = DF_1.loc[DF_1['DEPT'].isin(dept)].reset_index(drop=True)
    else:
        title_dept = "전체 학과의 교과목"
        DF_1 = DF_1.reset_index(drop=True)
        
    main_value = "MM_per_subj" # SCORE_NUM or MM_per_subj
    result_1 = DF_1.groupby(['DEPT', 'SUBJ', 'EMP', 'DIV'])[main_value].agg(
        MEAN='mean',
        ).reset_index().sort_index()
    result_1['MEAN'] = result_1['MEAN'].round(4)
    
    result_2 = DF_1.groupby(['DEPT','SUBJ', 'EMP'])['EMP'].size().reset_index(name='CNT').sort_index()
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
    for name, g_df in result.groupby(['DEPT','SUBJ', 'DIV'])[['CNT','EMP', 'MEAN']]:
        g_df = g_df.sort_values('EMP')
        if len(g_df) != 1:
            if (g_df.iloc[0,-3] > 2) & (g_df.iloc[1,-3] > 2):
                row = list(name) + [g_df.iloc[0,-1], g_df.iloc[1,-1]] + [g_df.iloc[0,-1] - g_df.iloc[1,-1]]
                rows_list.append(row)

    rows = sorted(rows_list,key=lambda x : (-x[-1], -x[-3], x[-2]))

    EMP_df = pd.DataFrame(rows, columns=['DEPT', 'SUBJ', 'DIV', '미취업', '취업', 'GAP'])

    EMP_df_for_graph = EMP_df.copy()
    
    try:
        green_df_for_graph = EMP_df_for_graph.loc[EMP_df_for_graph['GAP'] == 0]
        reversal_point = EMP_df_for_graph.loc[EMP_df_for_graph['GAP'] < 0].index[0]
        st.session_state['df_not_error'] = True
    except:
        for tab in tabs:
            with tab:
                st.error("선택된 학과(들)은 교과목 성적 정보를 취업/미취업으로 분류할 수 없습니다.\n* 데이터 부족\n* 취업/미취업 최소 구분 인원 수(각 2명씩) 충족 못함.")
        st.session_state['df_not_error'] = False
        return result
    
    dept_join_subj = EMP_df_for_graph['DEPT'] + "_" + EMP_df_for_graph['DIV'] + "_" + EMP_df_for_graph['SUBJ']
    cnt_dept = f"{title_dept} 개수 : {dept_join_subj.nunique()}개"
    
    EMP_df_for_graph = make_pie_graph(EMP_df_for_graph)

    st.session_state['cnt_dept'] = cnt_dept
    
    cols_list = [['미취업', '취업'], ['미취업_2', '취업_2']]
    for cdx, cols in enumerate(cols_list):
        fig = px.scatter(EMP_df_for_graph[cols], 
                        trendline="lowess", 
                        text = dept_join_subj,
                        trendline_options=dict(frac=0.25 if dept else 0.05),
                        title=f"{title_dept}\n취업VS미취업")
                
        fig.update_layout(
            width=1800,
            height=800,
            hovermode="x unified",
            margin=dict(l=40, r=40, t=40, b=40))      

        fig.update_traces(textfont=dict(color='rgba(0,0,0,0)'))

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

        fig.layout.xaxis.title.text = "교과목 인덱스"
        fig.layout.yaxis.title.text = "교과목 평균평점" if cdx == 0 else "교과목 정규화점수"
        
        st.session_state['figs'][cdx+2] = fig
        
    return result

all_dept_list = list(DF.DEPT.unique())
all_dept_list.sort()

st.header("학과 선택")
dept = st.multiselect(" ", all_dept_list, label_visibility='collapsed')
start = st.button("분석 시작")  
st.write("* 전체 학과로 검색하려면 공란으로 검색하세요.")
st.write("* 소요시간 : 5초 내외")

st.divider()
if 'button' not in st.session_state:
    st.session_state['button'] = False
if 'result' not in st.session_state:
    st.session_state['result'] = pd.DataFrame()
if 'figs' not in st.session_state:
    st.session_state['figs'] = [0,0,0,0]
if 'cnt_dept' not in st.session_state:
    st.session_state['cnt_dept'] = ""
if 'radio_state' not in st.session_state:
    st.session_state['radio_state'] = schl
if 'df_not_error' not in st.session_state:
    st.session_state['df_not_error'] = True

    
tabs = st.tabs(["**기본 그래프**", "**평균평점 세부 그래프**", "**정규화점수 세부 그래프**"])
if schl == st.session_state.radio_state:
    if start:
        st.session_state['button'] = True
        result = tmp(DF_1 = DF, dept=dept, scaled=True, tabs = tabs, select_term=20)
        st.session_state['result'] = result

    if st.session_state['button'] & st.session_state['df_not_error']:
        with tabs[0]:
            c1, _ = st.columns(2)
            with c1:
                select_term = st.slider("x축 개수 지정", 10, 50, 20)
            make_graphs(df=st.session_state['result'], title="과목", scaled=True, select_term=select_term, tabs = tabs)
            st.plotly_chart(st.session_state['figs'][0])
            st.subheader(st.session_state['cnt_dept'])
        with tabs[1]:
            st.plotly_chart(st.session_state['figs'][2])
            st.plotly_chart(st.session_state['figs'][1])
            st.subheader(st.session_state['cnt_dept'])
        with tabs[2]:
            st.plotly_chart(st.session_state['figs'][3])
            st.plotly_chart(st.session_state['figs'][1])
            st.subheader(st.session_state['cnt_dept'])
else:
    st.session_state['button'] = False
        
st.session_state.radio_state = schl