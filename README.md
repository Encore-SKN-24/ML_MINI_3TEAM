# ML_MINI_3TEAM
# 🌍# 🎹 [그럴 수 ~ 명있지]: 술과 기대수명의 착시에서 출발해, 오래 사는 진짜 이유를 찾다.
> WHO 데이터 EDA 프로젝트
---

## 1. 프로젝트 개요

### 1-1. 프로젝트 주제
**기계학습 방법론을 활용한 기대수명 영향 요인 분석 및 예측**

본 프로젝트는 WHO 국가별 보건 및 사회경제 데이터를 기반으로  
기대수명에 영향을 미치는 주요 요인을 분석하고,  
여러 회귀 모델을 비교하여 최적의 예측 모델을 도출하는 것을 목표로 한다.

---

### 1-2. 주제 선정 배경

데이터 탐색 과정에서  
**“알코올 소비량이 많으면 기대수명이 낮을까?”** 라는 단순 가설에서 출발하였다.

그러나 실제 데이터를 확인한 결과,

- GDP가 높은 국가일수록 알코올 소비량이 많지만
- 동시에 기대수명도 높은 경우가 존재하였다.

즉, 기대수명은 단일 요인이 아닌

- 경제 수준(GDP)
- 교육 수준(Schooling)
- 질병 발생률(HIV/AIDS 등)
- 예방접종률
- 보건 환경

등 다양한 요인이 복합적으로 작용하는 지표임을 확인하였다.

따라서 본 프로젝트에서는  
**다양한 보건·사회경제 지표를 활용하여 기대수명을 예측하고,  
어떤 요인이 중요한지 머신러닝 모델을 통해 분석**하고자 한다.
---
👥 팀 소개

| 성함 | GitHub |
| :---: | :---: |
| **권민제** | [![GitHub](https://img.shields.io/badge/min3802-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/min3802) |
| **문성준** | [![GitHub](https://img.shields.io/badge/dal--sj-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/dal-sj) |
| **전윤우** | [![GitHub](https://img.shields.io/badge/Yunu--Jeon-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Yunu-Jeon) |
| **정준하** | [![GitHub](https://img.shields.io/badge/junhaj27--jpg-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/junhaj27-jpg) |
| **최하진** | [![GitHub](https://img.shields.io/badge/hun668486-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/hun6684) |
---

## 2. 데이터셋 개요

### 2-1. 데이터 출처
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

### 2-2. 데이터 구성
- 단위: 국가별 연도별 보건 및 사회경제 지표
- Target 변수: **Life expectancy (기대수명)**

### 2-3. 주요 변수

| 변수 | 설명 |
|------|------|
| GDP | 1인당 국내총생산 |
| Schooling | 평균 교육 연수 |
| HIV/AIDS | HIV 감염률 |
| BMI | 평균 체질량지수 |
| Polio | 소아마비 예방접종률 |
| Diphtheria | 디프테리아 예방접종률 |
| Adult Mortality | 성인 사망률 |
| Alcohol | 알코올 소비량 |
| Life expectancy | 기대수명 (Target) |

---

## 3. 데이터 전처리 과정
---

## 3. 🛠 기술 스택
| 분류 | Stack |
| :--- | :--- |
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white) |
| **Visualization** | ![Seaborn](https://img.shields.io/badge/Seaborn-4479A1?style=for-the-badge&logo=Python&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-ffffff?style=for-the-badge&logo=Matplotlib&logoColor=black) |
| **Tool** | ![VSCode](https://img.shields.io/badge/VS_Code-007ACC?style=for-the-badge&logo=Visual-Studio-Code&logoColor=white) ![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=Git&logoColor=white) |
---
### 3-1. 불필요한 변수 제거

모델 학습에 직접적인 영향을 주지 않는 식별 변수 제거

제거된 변수:
- `country`
- `iso_code`
- `region`

→ 순수한 **수치형 보건·경제 지표만 사용**

---

### 3-2. 결측치 및 데이터 정리
- 결측치 확인 및 처리
- 이상치 분포 확인
- 수치형 변수만 모델 입력에 사용

---

### 3-3. 최종 학습 데이터
- 입력 변수: 보건 및 사회경제 수치 지표
- 목표 변수: 기대수명

---

## 4. 모델링 및 성능 비교

본 프로젝트에서는 **3가지 회귀 모델**을 적용하여 성능을 비교하였다.

### 사용 모델
1. Linear Regression
2. Random Forest Regressor
3. XGBoost Regressor

---

### 4-1. Random Forest 성능 (튜닝 전)

      MSE    RMSE     MAE   MSLE   RMSLE      R2


---

### 4-2. Random Forest 성능 (튜닝 후)

      MSE    RMSE     MAE   MSLE   RMSLE      R2


#### 성능 개선 결과
- RMSE: **2.13 → 2.00**
- R²: **0.9484 → 0.9542**

→ 하이퍼파라미터 튜닝을 통해 예측 성능 개선 확인

---

### 4-3. 모델 비교 요약

| 모델 | 특징 | 성능 경향 |
|------|------|-----------|
| Linear Regression | 단순, 해석 용이 | 기준 성능 |
| Random Forest | 비선형 관계 반영 | 높은 안정적 성능 |
| XGBoost | 부스팅 기반 고성능 모델 | 최고 성능 기대 |

최종적으로  
**비선형 관계를 잘 반영하는 트리 기반 모델이 더 높은 예측 성능**을 보였다.

---

## 5. 실제 예측 결과

- 모델은 다양한 보건·경제 지표를 종합적으로 반영하여 기대수명을 예측
- 단일 변수보다 **복합 지표 기반 예측이 더 정확함**을 확인

### 주요 영향 변수
- 성인 사망률
- HIV/AIDS 감염률
- 교육 수준
- 예방접종률
- GDP

---

## 6. 프로젝트 기대 효과

### 6-1. 정책적 활용
- 국가별 보건 정책 수립 시 핵심 영향 요인 파악 가능
- 의료·교육 투자 우선순위 설정에 활용

### 6-2. 데이터 기반 의사결정
- 단일 지표가 아닌 복합 지표 기반 건강 수준 평가
- 보건 환경 개선 전략 수립 가능

### 6-3. 머신러닝 적용 사례
- 실제 보건 데이터를 활용한 회귀 모델 적용 사례 제시
- 모델 비교 및 하이퍼파라미터 튜닝 과정 경험

---

## 7. 프로젝트 구조



---

## 8. 결론

- 기대수명은 단일 요인이 아닌 **복합적인 보건·경제 요인의 결과**임을 확인
- 선형 모델보다 **트리 기반 모델(Random Forest, XGBoost)**이 더 높은 성능을 보임
- 하이퍼파라미터 튜닝을 통해 모델 성능을 추가 개선 가능함을 확인

---

## 9. 향후 개선 방향

- 국가 그룹(선진국/개도국)별 모델 분리 학습
- 시계열 기반 기대수명 예측 모델 구축
- SHAP 기반 변수 중요도 해석 추가

