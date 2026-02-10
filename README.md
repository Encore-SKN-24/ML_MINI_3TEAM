# ML_MINI_3TEAM
## 🌍 [그럴 수 ~ 명있지]: 술과 기대수명의 착시에서 출발해, 오래 사는 진짜 이유를 찾다.
> WHO 데이터 EDA 프로젝트 ( 이거 수정 )
---
## 1. 👥 팀 소개

| 성함 | GitHub |
| :---: | :---: |
| **권민제** | [![GitHub](https://img.shields.io/badge/min3802-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/min3802) |
| **문성준** | [![GitHub](https://img.shields.io/badge/dal--sj-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/dal-sj) |
| **전윤우** | [![GitHub](https://img.shields.io/badge/Yunu--Jeon-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Yunu-Jeon) |
| **정준하** | [![GitHub](https://img.shields.io/badge/junhaj27--jpg-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/junhaj27-jpg) |
| **최하진** | [![GitHub](https://img.shields.io/badge/hun668486-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/hun6684) |

---
[
## 1. 프로젝트 개요
써야함

### 1-1. 프로젝트 주제
**기계학습 방법론을 활용한 기대수명 영향 요인 분석 및 예측**

본 프로젝트는 WHO 국가별 보건 및 사회경제 데이터를 기반으로  
기대수명에 영향을 미치는 주요 요인을 분석하고,  
여러 회귀 모델을 비교하여 최적의 예측 모델을 도출하는 것을 목표로 한다.
---

]
## 2. 📋 프로젝트 개요

### 💬 Project Story
“술은 건강에 좋지 않다고 하는데, 술을 많이 마시는 나라의 사람들은 실제로 어떨까?”라는  
가벼운 호기심에서 분석을 시작했습니다.

처음에는 **알코올 소비량**과 **기대수명**이 어느 한쪽으로 뚜렷한 관계를 보일 것이라 예상했지만,  
GDP가 높은 국가일수록 알코올 소비량이 많으면서도 기대수명이 높은 경우가 존재했고,
변수 간 관계는 일관되게 강한 상관관계를 보이지는 않았습니다.

이 과정에서 기대수명은 단순히 술 소비나 경제 수준만으로 설명되기 어렵고,  
**보건 환경, 교육 수준, 질병 관리, 예방접종 등 다양한 요인이 함께 얽혀 있는 지표**임을 확인했습니다.  
따라서 본 프로젝트는 WHO 기대수명 데이터를 활용해  
**사람들이 더 오래 사는 데 영향을 주는 요인(Key Factors)들이 무엇인지**를 탐색하는 것을 목표로 합니다.


### 📚 데이터 출처 (Data Sources)

| 분석 지표 | 제공 기관 | 내용 | 데이터 소스 (URL) |
| :--- | :---: | :--- | :--- |
| **메인 데이터셋** | Kaggle | Raw data | https://www.kaggle.com/datasets/<br>kumarajarshi/life-expectancy-who |
| **기대 수명** | WHO | 종속 변수 (Target) | https://www.who.int/data/gho/data/<br>indicators/indicator-details/GHO/<br>life-expectancy-at-birth-(years) |
| **교육 연한** | UNDP | 교육 수준 (Schooling) | https://hdr.undp.org/data-center/<br>documentation-and-downloads |
| **GDP (1인당)** | World Bank | 국가 경제 지표 | https://data.worldbank.org/indicator/<br>NY.GDP.PCAP.CD?most_recent_<br>year_desc=true |
| **알코올 소비량** | WHO | 성인 1인당 소비량 | https://www.who.int/data/gho/data/<br>indicators/indicator-details/GHO/<br>alcohol-recorded-per-capita-(15-) |
| **B형 간염 접종률** | WHO/UNICEF | 면역 시스템 지표 | https://www.who.int/data/gho/<br>data/indicators/indicator-details/GHO/hepatitis-b-(hepb3)<br>-immunization-coverage-among-1-year-olds-(-) |
---

## 3. 🛠 기술 스택
| 분류 | Stack |
| :--- | :--- |
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white) |
| **Visualization** | ![Seaborn](https://img.shields.io/badge/Seaborn-4479A1?style=for-the-badge&logo=Python&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-ffffff?style=for-the-badge&logo=Matplotlib&logoColor=black) |
| **Tool** | ![VSCode](https://img.shields.io/badge/VS_Code-007ACC?style=for-the-badge&logo=Visual-Studio-Code&logoColor=white) ![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=Git&logoColor=white) |
---

## 4. ⚙️ 데이터 전처리 (Preprocessing)
데이터의 무결성을 위해 `isna().sum()` 탐색 후 다음과 같은 **단계별 결측치 처리 로직**을 수립했습니다.

<img width="402" height="611" alt="결측치" src="https://github.com/user-attachments/assets/96afa0a9-1415-407a-b214-d104192c3370" />

1. **데이터 정화**: 여러 항목에 걸쳐 결측치가 너무 많은 국가(4개 컬럼 이상 비어있는 경우)는 분석의 왜곡을 방지하기 위해 가장 먼저 분석 대상에서 제외했습니다.
2. **시간적 보간**: 한 국가의 데이터 중 세로(연도) 방향으로 비어있는 경우, 해당 컬럼의 **전후년 평균값**으로 대치했습니다.
3. **공간적 보간**: 전후 데이터가 없는 경우, 해당 국가가 속한 **지역(Region)의 평균값**을 활용하여 지역적 특성을 반영했습니다.


## 5. 📊 수행 결과

### 🧪 1. 상관관계 분석 (Heatmap)
<img width="1050" height="998" alt="image" src="https://github.com/user-attachments/assets/a77af736-18b4-4c70-a110-d4252e0f4d3c" />


* **핵심 인사이트**: 히트맵 분석 결과, 예상과 달리 **알코올 소비량은 기대수명과 유의미하게 높은 상관관계를 보이지 않았습니다.** 이는 술 소비량 자체가 수명을 결정짓는 단일 요인이 아님을 입증합니다.


## 🎯 최종 결론: 
---

## 6. 💬 한 줄 회고
* **[권민제]**:
* **[문성준]**:
* **[전윤우]**:
* **[정준하]**:
* **[최하진]**: 
