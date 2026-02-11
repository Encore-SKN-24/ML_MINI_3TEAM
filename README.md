# ML_MINI_3TEAM
## 🌍 [그럴 수 ~ 명있지]: 데이터로 오래 사는 진짜 이유를 찾다.
> WHO 데이터를 기반으로 기대수명을 결정짓는 구조적 요인을 분석한 데이터 프로젝트

## 1. 팀 소개

| 이름 | GitHub |
| :---: | :---: |
| **권민제** | [![GitHub](https://img.shields.io/badge/min3802-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/min3802) |
| **문성준** | [![GitHub](https://img.shields.io/badge/dal--sj-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/dal-sj) |
| **전윤우** | [![GitHub](https://img.shields.io/badge/Yunu--Jeon-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Yunu-Jeon) |
| **정준하** | [![GitHub](https://img.shields.io/badge/junhaj27--jpg-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/junhaj27-jpg) |
| **최하진** | [![GitHub](https://img.shields.io/badge/hun6684-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/hun6684) |

## 2. 프로젝트 개요

### 2-1. Project Story: 무엇이 인간의 수명을 결정짓는가?

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

### 2-2. 데이터 출처 (Data Sources)

| 분석 지표 | 제공 기관 | 내용 | 데이터 소스 (URL) |
| :--- | :---: | :--- | :--- |
| **메인 데이터셋** | Kaggle | Raw data | https://www.kaggle.com/datasets/<br>kumarajarshi/life-expectancy-who |
| **기대 수명** | WHO | 종속 변수 (Target) | https://www.who.int/data/gho/data/<br>indicators/indicator-details/GHO/<br>life-expectancy-at-birth-(years) |
| **교육 연한** | UNDP | 교육 수준 (Schooling) | https://hdr.undp.org/data-center/<br>documentation-and-downloads |
| **GDP (1인당)** | World Bank | 국가 경제 지표 | https://data.worldbank.org/indicator/<br>NY.GDP.PCAP.CD?most_recent_<br>year_desc=true |
| **알코올 소비량** | WHO | 성인 1인당 소비량 | https://www.who.int/data/gho/data/<br>indicators/indicator-details/GHO/<br>alcohol-recorded-per-capita-(15-) |
| **B형 간염 접종률** | WHO/UNICEF | 면역 시스템 지표 | https://www.who.int/data/gho/<br>data/indicators/indicator-details/GHO/hepatitis-b-(hepb3)<br>-immunization-coverage-among-1-year-olds-(-) |

## 3. 기술 스택
| Category | Libraries / Tools |
| :--- | :--- |
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) |
| **Data Processing** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge) ![Seaborn](https://img.shields.io/badge/Seaborn-5A9BD5?style=for-the-badge) |
| **Machine Learning** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![XGBoost](https://img.shields.io/badge/xgboost-%2324272A.svg?style=for-the-badge&logo=xgboost&logoColor=white) |
| **HyperParameter Tuning** | ![Optuna](https://img.shields.io/badge/Optuna-%235062A1.svg?style=for-the-badge&logo=target&logoColor=white)
| **Collaboration Tool** | ![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white) |


## 4.  데이터 전처리 (Preprocessing)
### 4-1. heatmap을 통한 상관관계 분석
<img width="1050" height="972" alt="image" src="https://github.com/user-attachments/assets/6eb5cf23-ee80-499f-b240-83b3aa792436" />

* **핵심 인사이트**: 히트맵 분석 결과, 예상과 달리 **알코올 소비량은 기대수명과 유의미하게 높은 상관관계를 보이지 않았습니다.** 이는 술 소비량 자체가 수명을 결정짓는 단일 요인이 아님을 입증합니다.

### 4-2. 결측치 및 이상치 처리
#### 4-2-1. 전체적인 과정 소개
- **데이터 정화**: 여러 항목에 걸쳐 결측치가 너무 많은 국가(4개 컬럼 이상 비어있는 경우)는 분석의 왜곡을 방지하기 위해 가장 먼저 분석 대상에서 제외했습니다.
- **시간적 보간**: 한 국가의 데이터 중 세로(연도) 방향으로 비어있는 경우, 해당 컬럼의 **전후년 평균값**으로 대치했습니다.
- **공간적 보간**: 전후 데이터가 없는 경우, 해당 국가가 속한 **지역(Region)의 평균값**을 활용하여 지역적 특성을 반영했습니다.

#### 4-2-2. 기술통계량 비교
- Raw Data
<img width="2437" height="324" alt="image" src="https://github.com/user-attachments/assets/1ffa46f7-3609-415a-9dd0-16ab1e0f562b" />
- Cleaned Data
<img width="2660" height="534" alt="image" src="https://github.com/user-attachments/assets/3bfcd50a-311b-48bf-be93-e0f21eaa3e8d" />

## 5. 사용한 모델과 학습 성과
### 5-1. RandomForestRegression
#### 5-1-1. 데이터
feature 데이터를 구성할 때 target에 해당하는 ‘Life_expectancy’와 범주형 자료들과 target과 직접적인 관련이 있는 ‘HIV_AIDS’를 제외했습니다.

#### 5-1-2. Optuna 적용 전
<img width="436" height="67" alt="image" src="https://github.com/user-attachments/assets/c3d4f9d7-4e78-41d1-b93a-ca85db2c87fe" />

#### 5-1-2. Optuna 적용 전 학습 곡선
- max_depth = 8 이후로 Training Score와 Cross-validation Score의 RMSE의 차이가 커져 과대적합이 발생했습니다.
<img width="554" height="455" alt="image" src="https://github.com/user-attachments/assets/c82341f5-3bfa-47e4-8706-ab68bbe8045d" />
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/53331494-ff50-465b-873b-43e2b3947293" />

#### 5-1-3. RandomForestRegression에 Optuna를 적용한 결과
<img width="213" height="111" alt="image" src="https://github.com/user-attachments/assets/3eee1ab1-a09e-4fbe-9c25-ef10402441dd" />
<img width="435" height="86" alt="image" src="https://github.com/user-attachments/assets/f2a6b5ed-c432-4235-abc3-b3f495a21128" />

#### 5-1-4. 시각화
<img width="1089" height="789" alt="image" src="https://github.com/user-attachments/assets/c417ef8d-a8b4-4fdf-8875-4f8c068c6cc2" />

### 5-2. XGBoost
#### 5-2-1. 데이터
평균 기대 수명 예측과 관련이 없거나 범주형 데이터인 country, iso_code, region, year. 그리고 너무 직접적인 관련이 있는 hiv_aids은 제거하였다.
#### 5-2-2. 최적화 전
<img width="441" height="76" alt="image" src="https://github.com/user-attachments/assets/47b870ad-77a3-41c2-9c80-6754ae4a5f49" />

#### 5-2-3. trial.suggest_int('max_depth', 8, 11)
<img width="318" height="161" alt="image" src="https://github.com/user-attachments/assets/741d91d3-4ae6-468d-bdec-fccacd5fc288" />
<img width="438" height="72" alt="image" src="https://github.com/user-attachments/assets/a37e24ed-9965-481b-a44f-ad0ca497ed5f" />

#### 5-2-4. 기대수명에 영향을 미치는 Features에 대한 시각화 결과
<img width="1122" height="697" alt="image" src="https://github.com/user-attachments/assets/027ec963-e4ec-4ee5-b4c0-b2cf0a015cbc" />

## 6. 🎯 최종 결론:
본 프로젝트에서는 WHO 국가별 보건·사회경제 데이터를 활용하여 기대수명을 예측하고, 주요 영향 요인을 분석하였다. 여러 회귀 모델을 비교한 결과, RandomForestRegressor와 XGBoost 두 모델 모두에서 Schooling(교육연한)이 가장 중요한 변수 중 하나로 도출되었다.

이는 기대수명이 단순히 의료 지표나 질병 요인만으로 결정되는 것이 아니라,
교육 수준과 같은 사회구조적 요인이 장기적으로 건강 상태와 삶의 질에 큰 영향을 미친다는 점을 시사한다. 교육 수준이 높을수록 건강 정보 접근성이 증가하고, 예방 중심의 생활 습관 형성, 의료 서비스 활용 능력 향상, 안정적인 경제 활동 등 다양한 긍정적 효과가 누적되어 기대수명 증가로 이어질 가능성이 크다

## 7. 💬 한 줄 회고
* **[권민제]**:  데이터 간의 실제 상관관계는 모델을 직접 구동해 봐야만 선명해진다는 것을 깨달았고, 이에 맞춰 적절한 모델을 선택하는 안목의 중요성을 실감했습니다. 또한 과적합 방지부터 스코어 관리까지 모든 과정이 정교하게 맞물려야만 정밀한 모델이 완성된다는 것을 배운 만큼, 다음 프로젝트에서는 전 과정의 정합성을 고려한 모델을 구축하는 데 집중하겠습니다.
* **[문성준]**: Optuna 최적화와 K-Fold 검증까지 이어지는 과정을 수행하여 학습의 질을 높일 수 있었다. 그 과정에서 학습곡선 출력을 통해 과적합 여부를 판단하고자 해보았다. 다만, 정작 과적합 여부를 판단하는 명확한 근거나 기준 설정이 없어 무식하게 학습 횟수를 늘려 우선 test set의 rmse를 낮추는 것에만 집중하였다. 추후 프로젝트에서는 과적합 방지를 위해 데이터 수를 늘린다던지, 관련 하이퍼파라미터의 설정은 직접 하는 식으로 더 효율적인 방법을 시도해보겠다.
* **[전윤우]**: 수업시간에 다룬 머신러닝의 이론과 실습을 넘어 방대한 데이터로 Machine Learning 실습을 하는 것은 쉽지 않았습니다. 전처리 과정에서는 이상치나 결측치가 많았고, 모델 학습 및 평가에 있어서는 기대한 만큼 모델의 성능이 좋지 않다거나 하는 문제가 있었습니다. 그러나 팀원들과 힘을 합쳐 최선의 성능을 내기 위해 노력을 하였다는 점에서 의의가 있었습니다. 2차 프로젝트 때는 양적인 측면과 질적인 측면을 모두 고려한 데이터셋을 찾고 더 나은 모델 학습 및 평가 방법론을 적용해볼 수 있게 노력할 것입니다.
* **[정준하]**: 여러 회귀 모델을 직접 적용하고 전처리 과정을 반복하며 성능을 비교하는 과정에서, 기대수명은 단일 변수로 설명되는 것이 아니라 경제·보건·교육 등 여러 요인이 함께 작용하는 복합적인 지표임을 데이터로 확인할 수 있었던 의미 있는 경험이었다.
* **[최하진]**: 모델 성능을 개선하는 과정에서 어려움이 있었지만, 모델을 직접 적용하고 결과를 비교해보는 과정에서 전처리, 특성 스케일링, 하이퍼파라미터 설정이 성능에 얼마나 큰 영향을 미치는지를 알아가며 모델의 구조와 동작 원리를 이해하게 되었다. 다음에는 다른 모델들도 함께 비교해보고, 보다 정밀한 하이퍼파라미터 튜닝을 중심으로 성능 향상을 시도해보고 싶다.
