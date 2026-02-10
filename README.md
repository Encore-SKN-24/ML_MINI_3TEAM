# ML_MINI_3TEAM
## 🌍 [그럴 수 ~ 명있지]: 데이터로 오래 사는 진짜 이유를 찾다.
> WHO 데이터를 기반으로 기대수명을 결정짓는 구조적 요인을 분석한 데이터 프로젝트
---
## 1. 👥 팀 소개

| 성함 | GitHub |
| :---: | :---: |
| **권민제** | [![GitHub](https://img.shields.io/badge/min3802-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/min3802) |
| **문성준** | [![GitHub](https://img.shields.io/badge/dal--sj-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/dal-sj) |
| **전윤우** | [![GitHub](https://img.shields.io/badge/Yunu--Jeon-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Yunu-Jeon) |
| **정준하** | [![GitHub](https://img.shields.io/badge/junhaj27--jpg-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/junhaj27-jpg) |
| **최하진** | [![GitHub](https://img.shields.io/badge/hun6684-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/hun6684) |

---

## 2. 📋 프로젝트 개요

### 🧬 Project Story: 무엇이 인간의 수명을 결정짓는가?

“**술을 많이 마시는 나라는 정말 수명이 짧을까?**”  
이 단순한 질문에서 출발한 분석은, 예상보다 훨씬 복잡한 데이터의 연결고리를 드러냈습니다.

초기 가설과 달리, 실제 데이터에서는 **경제 수준이 높으면서 알코올 소비량도 많고, 동시에 기대수명까지 높은 국가들**이 존재했습니다.  
이는 단일한 생활 습관만으로 인간의 수명을 단순한 설명으로는 충분하지 않다는 사실을 확인했습니다.

이에 본 프로젝트는 WHO 기대수명 데이터를 활용해, 개인의 선택을 넘어 기대수명의 형성에 결정적으로 작용하는 핵심 요인을 분석했습니다.

분석 결과, 기대수명은 개인의 생활 습관만으로 설명되기보다는

- **교육 수준과의 강한 상관관계(0.72)**
- 히트맵에서 볼 수 있는걸 추가하기

등 사회적·제도적 요인이 있다는 것을 확인했습니다.

더 나아가, 이러한 복합 요인들이 실제로 **기대수명을 얼마나 정확히 설명하고 예측할 수 있는지**를 검증하기 위해  **Random Forest**와 **XGBoost** 모델을 구축했습니다.

비선형적 특성을 지닌 데이터를 머신러닝으로 학습시켜,  **수명 예측에 가장 결정적인 변수는 무엇인지** 살펴보는 것을 본 프로젝트의 최종 목표로 삼았습니다.

---

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
### 🧪 1. 상관관계 분석 (Heatmap)
<img width="1050" height="998" alt="image" src="https://github.com/user-attachments/assets/a77af736-18b4-4c70-a110-d4252e0f4d3c" />

* **핵심 인사이트**: 히트맵 분석 결과, 예상과 달리 **알코올 소비량은 기대수명과 유의미하게 높은 상관관계를 보이지 않았습니다.** 이는 술 소비량 자체가 수명을 결정짓는 단일 요인이 아님을 입증합니다.

1. **데이터 정화**: 여러 항목에 걸쳐 결측치가 너무 많은 국가(4개 컬럼 이상 비어있는 경우)는 분석의 왜곡을 방지하기 위해 가장 먼저 분석 대상에서 제외했습니다.
2. **시간적 보간**: 한 국가의 데이터 중 세로(연도) 방향으로 비어있는 경우, 해당 컬럼의 **전후년 평균값**으로 대치했습니다.
3. **공간적 보간**: 전후 데이터가 없는 경우, 해당 국가가 속한 **지역(Region)의 평균값**을 활용하여 지역적 특성을 반영했습니다.


## 5. 📊 수행 결과


## 6. 🎯 최종 결론: 
---

## 7. 💬 한 줄 회고
* **[권민제]**:
* **[문성준]**:
* **[전윤우]**:
* **[정준하]**:
* **[최하진]**: 
