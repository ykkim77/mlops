# 6주차 실습: KFP로 만드는 경량 MLOps 파이프라인

> 「AI플랫폼 06. MLOps 파이프라인 구축 실습」 강의자료 연계 실습서
> 대상: KFP v2 백엔드가 설치된 Kubeflow 환경 / GPU 불필요
> 작성·컴파일 환경: Kubeflow Notebook의 `ykkim77/kfp-notebook:py310-kfp250-v1` 이미지 사용
> 기본 비교 방식: 같은 Experiment에 Run을 모아 Kubeflow UI에서 비교함

## 1. 실습 목표와 범위

- KFP SDK의 `@dsl.component`, `@dsl.pipeline` 문법 익힘
- 데이터 전처리 → 모델 학습 → 모델 평가를 독립된 컨테이너 단계로 구성함
- Dataset·Model·Metrics 아티팩트 전달 및 실행 이력 추적을 이해함
- 같은 Experiment에서 여러 Run의 성능과 비용 지표를 비교함
- 데이터 누수 방지, 실행 재현성, 품질 기준, 자원 제한 등 MLOps 관점을 익힘

이 실습은 학습 파이프라인 자동화와 실험 추적에 집중함. 운영 배포, CI/CD, 운영 데이터 드리프트 감시까지 구현하는 전체 MLOps 시스템은 아님. 파이프라인을 한 번 실행한 것만으로 자동 재학습 체계를 완성했다고 보지는 않음.

## 2. 구성 및 자원 사용

| 항목 | 실습 구성 |
|---|---|
| 데이터 | scikit-learn 내장 Iris: 150개 샘플, 4개 특성, 3개 클래스 |
| 기본 분할 | 학습 105개 / 평가 45개, 계층화 분할 |
| 모델 | RandomForestClassifier, CPU 한 스레드 |
| 단계 | preprocess → train → evaluate |
| 핵심 실행 | 같은 Experiment에서 A·B·C 3개 Run을 순차 실행 |
| 확장 실행 | D: 재현성 / E: 품질 기준 / F·G: 데이터 분할 비교 |
| 단계별 사용자 컨테이너 요청 | CPU 100m, 메모리 256Mi |
| 단계별 사용자 컨테이너 상한 | CPU 1코어, 메모리 1Gi |
| 별도 도입 서비스 | 없음: MLflow·Katib·KServe·분산학습 미사용 |

- 원본 데이터가 패키지에 포함되어 데이터 다운로드 불필요
- 학습 연산은 매우 작지만 Pod 시작, 이미지 다운로드, 패키지 설치가 전체 실행 시간의 대부분을 차지할 수 있음
- 메모리 상한은 학습 데이터보다 패키지 설치 및 Python 실행을 고려한 값임
- 위 자원 설정은 사용자 작업 컨테이너 기준이며, KFP 런처·사이드카·플랫폼 서비스까지 포함한 전체 클러스터 사용량은 아님
- Kubeflow 자체의 상시 자원 사용량은 별도로 필요함
- 모든 Run을 동시에 실행하지 않고 앞선 Run 종료 후 다음 Run 실행함

## 3. MLOps 개념과 실습의 연결

| 개념 | 실습에서 확인할 내용 |
|---|---|
| 파이프라인 자동화 | 코드로 단계와 의존성을 정의하여 동일 흐름 반복 실행 |
| 실험 추적 | Experiment에 Run을 모으고 파라미터·메트릭·산출물 기록 |
| 재현성 | random_state, 데이터 분할, 패키지 버전을 고정하여 성능 재확인 |
| 데이터 누수 방지 | 먼저 분할하고 학습 데이터로만 정규화 통계 계산 |
| 모델 계보 | 학습 Dataset → Model → 평가 Metrics 연결을 Run 그래프에서 추적 |
| 품질 기준 | accuracy가 min_accuracy 이상인지 quality_pass로 기록 |
| 자원 관리 | n_jobs=1, 스레드 제한, CPU·메모리 상한 적용 |
| 모델 선택 | accuracy·F1뿐 아니라 시간·모델 크기를 함께 비교 |
| 환경 및 버전 관리 | SDK·라이브러리 버전 고정, 작성한 코드는 Git 관리 대상으로 이해 |

정규화는 강의자료의 전처리 설명을 익히기 위해 포함함. RandomForest처럼 트리 기반 모델은 일반적으로 표준화가 필수는 아님. 실제 서비스에서는 학습 때 사용한 전처리를 추론에도 동일하게 적용해야 하므로 모델 파일에 정규화 통계를 함께 저장함.

## 4. 사전 확인

### 4.1. 클러스터 및 대시보드

```bash
kubectl get nodes
kubectl get pods -n kubeflow
kubectl get pvc -n kubeflow
kubectl get sc
```

- 관련 서비스가 정상 동작하고 KFP 대시보드에 접속 가능한 상태에서 시작함
- KFP 저장소의 PVC는 소비 Pod 생성 후 Bound 상태인지 확인함
- local-path StorageClass는 실습에 사용 가능하며, 아티팩트 파일은 KFP가 설정한 아티팩트 저장소를 통해 전달됨
- 각 단계가 같은 로컬 파일 경로나 같은 worker에 있어야 한다는 뜻은 아님
- Python 기본 이미지 및 PyPI 패키지 접근이 작업 Pod에서 가능해야 함
- 강의자료의 Kind 환경 또는 기존 온프레미스 클러스터를 재사용함

필요 시 대시보드 연결 예시:

```bash
kubectl port-forward -n istio-system svc/istio-ingressgateway 8080:80
```

브라우저: <http://localhost:8080>. 원격 서버라면 로컬 PC에서 SSH 터널을 연결함. 이미 사용 가능한 대시보드 주소가 있으면 그 주소를 사용함.

### 4.2. 커스텀 Notebook 생성

1. Kubeflow 대시보드에서 자신의 사용자 namespace 선택함
2. Notebooks → New Notebook 또는 New Server 선택함
3. Notebook 이름을 `mlops-lab`으로 지정함
4. 이미지 항목에서 Custom image 선택 후 아래 주소 입력함

```text
ykkim77/kfp-notebook:py310-kfp250-v1
```

5. CPU 요청 0.5코어, 메모리 요청 1Gi, GPU 없음으로 설정함
6. 사용자별 Workspace PVC를 연결하고 Notebook 생성함
7. Ready 상태 확인 후 CONNECT로 JupyterLab 접속함
8. Launcher에서 `Python 3.10 - KFP 2.5.0` 커널을 선택하여 새 Notebook 생성함
9. 파일명을 `00_environment_check.ipynb`로 변경하고 다음 셀 실행함

```python
# 현재 Notebook 커널의 Python 및 KFP 버전 확인함
import sys
import kfp

print("Python:", sys.version)
print("KFP:", kfp.__version__)
assert sys.version_info[:2] == (3, 10), "Python 3.10 커널 선택 여부 확인"
assert kfp.__version__ == "2.5.0", "지정한 커스텀 이미지 사용 여부 확인"
```

- 커스텀 이미지에 실습 패키지가 포함되어 있으므로 별도 패키지 설치 과정 불필요
- JupyterLab의 Terminal에서 `mkdir -p ~/work/kfp-iris-lab` 실행함
- 파일 브라우저에서 `work/kfp-iris-lab` 폴더로 이동함
- File → New → Text File로 새 파일을 만들고 `iris_pipeline.py`로 이름 변경함
- 아래 전체 코드를 붙여넣고 Ctrl+S로 저장함
- `.ipynb`는 환경 확인용, `.py`는 KFP 컴포넌트 정의 및 컴파일용으로 사용함
- XSRF cookie 오류 발생 시 시크릿 창에서 재접속하여 확인하고 기존 접속 주소의 사이트 쿠키 정리함

### 4.3. Notebook 이미지와 파이프라인 작업 이미지 구분

| 구분 | 이미지 | 역할 |
|---|---|---|
| Notebook | `ykkim77/kfp-notebook:py310-kfp250-v1` | 코드 작성·버전 확인·YAML 컴파일 |
| 각 파이프라인 작업 | 코드에 지정한 `python:3.10-slim` | 전처리·학습·평가 실행 |

- Notebook에 설치된 패키지가 다른 작업 Pod에 자동 전달되는 것은 아님
- 기존 실행 성공 코드의 `base_image`와 `packages_to_install`을 유지함
- 작업 Pod에서 기본 이미지 다운로드와 PyPI 패키지 설치가 가능해야 함

## 5. 전체 파이프라인 코드

파일명: `iris_pipeline.py`

```python
# compiler: Python 파이프라인을 YAML로 변환함
# dsl: 컴포넌트·파이프라인 정의 기능 제공함
from kfp import compiler, dsl
# Input·Output: 아티팩트 입출력 방향을 표시하는 KFP 타입임
# Dataset·Model 등: 아티팩트의 논리적 종류이며 파일 형식을 강제하지 않음
from kfp.dsl import Input, Output, Dataset, Model, Metrics, ClassificationMetrics

# 컴포넌트 내부에서 사용하는 패키지는 컨테이너에도 설치되어야 함
# 버전 고정으로 작업 컨테이너의 라이브러리 환경 통일함
PACKAGES = [
    "numpy==1.26.4", "scipy==1.13.1", "scikit-learn==1.5.2",
    "joblib==1.4.2", "threadpoolctl==3.5.0",
]


@dsl.component(base_image="python:3.10-slim", packages_to_install=PACKAGES)
# 전처리 컴포넌트 정의함. 실제 함수 본문은 Run의 작업 컨테이너에서 실행됨
def preprocess(
    # 평가 데이터 비율을 숫자 파라미터로 입력받음
    test_size: float,
    # 난수 시드를 숫자 파라미터로 입력받음
    random_state: int,
    # 학습 데이터의 출력 아티팩트 선언함. 객체와 저장 경로는 KFP가 제공함
    train_data: Output[Dataset],
    # 평가 데이터의 출력 아티팩트 선언함
    test_data: Output[Dataset],
):
    import os
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size는 0.1~0.5 범위로 지정")
    # 패키지에 포함된 150개 Iris 샘플을 읽음. 별도 데이터 다운로드 불필요
    iris = load_iris()
    # X는 특성값, y는 정답 클래스임
    # stratify로 학습·평가 데이터의 클래스 비율 유지함
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=test_size,
        random_state=random_state, stratify=iris.target,
    )
    # 반드시 분할 후 학습 데이터에서만 평균/표준편차 학습: 데이터 누수 방지
    # 특성별 평균 0·표준편차 1로 변환할 정규화 객체 생성함
    scaler = StandardScaler()
    # 학습 데이터만 사용해 평균·표준편차를 구하고 변환함
    x_train = scaler.fit_transform(x_train)
    # 평가 데이터는 학습에서 얻은 통계로만 변환함
    x_test = scaler.transform(x_test)
    # 학습·평가 결과를 각각의 출력 파일로 저장함
    for artifact, x, y in [(train_data, x_train, y_train), (test_data, x_test, y_test)]:
        # KFP가 제공한 아티팩트 저장 경로의 부모 폴더 생성함
        os.makedirs(os.path.dirname(artifact.path), exist_ok=True)
        # 파일 객체 사용: np.savez가 .npz 확장자를 자동 추가하는 문제 방지
        with open(artifact.path, "wb") as f:
            # 배열·정규화 통계·클래스 이름을 압축된 NumPy 형식으로 저장함
            np.savez_compressed(
                f, X=x, y=y, mean=scaler.mean_, scale=scaler.scale_,
                labels=np.array(iris.target_names),
            )
        # 데이터의 출처·크기·분할 설정을 메타데이터로 기록함
        artifact.metadata.update({
            "dataset": "sklearn-iris", "rows": int(len(y)),
            "features": 4, "random_state": random_state,
            "test_size": test_size, "scaler_fit": "train-only",
        })
    print(f"train={len(y_train)}, test={len(y_test)}")


@dsl.component(base_image="python:3.10-slim", packages_to_install=PACKAGES)
# 학습 컴포넌트 정의함. Dataset을 받아 Model을 출력함
def train(
    # 전처리 작업이 만든 학습 데이터 아티팩트를 입력받음
    train_data: Input[Dataset],
    # RandomForest 트리 수를 입력받음
    n_estimators: int,
    # 각 트리의 최대 깊이를 입력받음
    max_depth: int,
    # 난수 시드를 숫자 파라미터로 입력받음
    random_state: int,
    # 모델 출력 아티팩트 선언함. model은 자유롭게 정하는 인자 이름임
    # model2로 변경 시 함수 내부 참조와 fitted.outputs의 키도 변경해야 함
    model: Output[Model],
):
    import os
    import time
    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    # 실습 중 과도한 자원 사용을 방지하도록 파라미터 범위 제한함
    if not 1 <= n_estimators <= 100:
        raise ValueError("경량 실습을 위해 n_estimators는 1~100으로 제한")
    if not 1 <= max_depth <= 10:
        raise ValueError("max_depth는 1~10으로 제한")
    # 입력 아티팩트 경로에서 전처리된 배열을 읽음
    with np.load(train_data.path, allow_pickle=False) as data:
        x, y = data["X"], data["y"]
        mean, scale = data["mean"], data["scale"]
        labels = data["labels"].tolist()
    # 실제 머신러닝 모델 객체를 생성함. 이 시점에는 아직 학습되지 않음
    # n_jobs=1로 학습에 사용하는 병렬 작업 수 제한함
    classifier = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=random_state, n_jobs=1,
    )
    # 학습 연산 시간 측정을 시작함. 설치·Pod 대기 시간은 포함하지 않음
    start = time.perf_counter()
    # 학습 특성 x와 정답 y로 모델 학습함
    classifier.fit(x, y)
    fit_seconds = time.perf_counter() - start
    # 추론에서도 동일 전처리를 재사용할 수 있도록 통계와 모델을 함께 보관
    # 학습한 모델과 정규화 통계 및 실험 정보를 하나의 묶음으로 구성함
    bundle = {
        "classifier": classifier, "scaler_mean": mean, "scaler_scale": scale,
        "labels": labels, "fit_seconds": fit_seconds,
        "n_estimators": n_estimators, "max_depth": max_depth,
        "random_state": random_state,
    }
    # 모델 산출물 저장 폴더 생성함
    os.makedirs(os.path.dirname(model.path), exist_ok=True)
    # 학습 모델 묶음을 파일로 저장함. Output[Model] 선언만으로 저장되지는 않음
    joblib.dump(bundle, model.path)
    # 알고리즘·하이퍼파라미터·라이브러리 버전을 모델 메타데이터에 기록함
    model.metadata.update({
        "algorithm": "RandomForestClassifier", "n_estimators": n_estimators,
        "max_depth": max_depth, "random_state": random_state,
        "sklearn_version": "1.5.2",
    })
    print(f"fit_seconds={fit_seconds:.6f}")


@dsl.component(base_image="python:3.10-slim", packages_to_install=PACKAGES)
# 평가 컴포넌트 정의함. 모델과 평가 데이터를 입력받아 지표를 출력함
def evaluate(
    # 학습 작업이 저장한 모델 아티팩트를 입력받음
    model: Input[Model],
    # 전처리 작업이 저장한 평가 데이터 아티팩트를 입력받음
    test_data: Input[Dataset],
    # 품질 통과 여부를 판단할 최소 정확도 입력받음
    min_accuracy: float,
    # Run 비교에 사용할 숫자 지표 출력 선언함
    metrics: Output[Metrics],
    # 분류 결과를 시각화할 혼동행렬 출력 선언함
    confusion: Output[ClassificationMetrics],
    # 보조 JSON 결과 파일 출력 선언함. 본 실습의 비교는 대시보드에서 수행함
    report: Output[Dataset],
# bool 반환값은 품질 통과 여부이며 파일 아티팩트가 아닌 출력 파라미터임
) -> bool:
    import json
    import os
    import numpy as np
    import joblib
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    if not 0.0 <= min_accuracy <= 1.0:
        raise ValueError("min_accuracy는 0~1 범위로 지정")
    # 신뢰할 수 있는 실습 모델 파일을 읽어 모델 객체 복원함
    bundle = joblib.load(model.path)
    with np.load(test_data.path, allow_pickle=False) as data:
        x, y = data["X"], data["y"]
    # x는 preprocess 단계에서 이미 정규화됨: 다시 정규화하지 않음
    # 정답을 알려주지 않고 평가 특성값으로 클래스를 예측함
    predicted = bundle["classifier"].predict(x)
    # 전체 샘플 중 정답 비율 계산함
    accuracy = float(accuracy_score(y, predicted))
    # 각 클래스의 F1 점수를 같은 비중으로 평균함
    macro_f1 = float(f1_score(y, predicted, average="macro", zero_division=0))
    # 실행 성공과 별개로 모델 품질 기준 통과 여부 판단함
    passed = accuracy >= min_accuracy
    # 성능·학습 시간·파일 크기·설정값을 비교용 지표로 모음
    values = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "train_seconds": float(bundle["fit_seconds"]),
        "model_size_kib": os.path.getsize(model.path) / 1024.0,
        "test_samples": int(len(y)),
        "n_estimators": int(bundle["n_estimators"]),
        "max_depth": int(bundle["max_depth"]),
        "quality_pass": int(passed),
    }
    # 모든 Run에서 동일한 지표 이름을 사용하여 비교 가능하도록 기록함
    for name, value in values.items():
        # Metrics 아티팩트의 메타데이터에 숫자 지표 기록함
        metrics.log_metric(name, float(value))
    # 행=실제 클래스, 열=예측 클래스인 혼동행렬 생성함
    # Iris 클래스 번호 0·1·2를 사용함
    matrix = confusion_matrix(y, predicted, labels=[0, 1, 2])
    # 클래스 이름과 행렬을 UI 시각화용으로 기록함
    confusion.log_confusion_matrix(bundle["labels"], matrix.tolist())
    # 결과와 실험 조건을 보조 보고서로 구성함
    result = {
        "metrics": values, "min_accuracy": min_accuracy,
        "random_state": bundle["random_state"],
        "confusion_matrix": matrix.tolist(), "labels": bundle["labels"],
    }
    os.makedirs(os.path.dirname(report.path), exist_ok=True)
    with open(report.path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    report.metadata["format"] = "json"
    # Run 상세 화면의 Logs 탭에서도 결과 확인 가능하도록 출력함
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # 품질 미달이어도 예외를 발생시키지 않으므로 Run은 정상 완료 가능함
    return passed


# 파이프라인 함수 등록함. 이 함수를 컴파일하여 DAG 생성함
@dsl.pipeline(name="iris-mlops-light", description="Iris 전처리-학습-평가 및 Run 메트릭 비교")
def iris_pipeline(
    # Run 생성 UI에서 변경할 수 있는 기본 파라미터 정의함
    n_estimators: int = 10,
    max_depth: int = 2,
    test_size: float = 0.3,
    random_state: int = 42,
    min_accuracy: float = 0.9,
):
    # 전처리 작업 객체를 생성하고 prep으로 참조함. 여기서 즉시 전처리하지 않음
    # 입력값이 Run 파라미터뿐이며 선행 작업 의존성이 없어 시작 작업이 됨
    prep = preprocess(test_size=test_size, random_state=random_state)
    # prep의 학습 출력을 train 입력으로 연결함 → prep 완료 후 train 실행됨
    # 같은 인자 이름이라 자동 연결되는 것이 아니라 outputs 참조로 명시 연결함
    fitted = train(
        train_data=prep.outputs["train_data"], n_estimators=n_estimators,
        max_depth=max_depth, random_state=random_state,
    )
    # fitted의 모델과 prep의 평가 데이터를 연결함 → 둘의 결과 준비 후 평가 실행됨
    evaluated = evaluate(
        model=fitted.outputs["model"], test_data=prep.outputs["test_data"],
        min_accuracy=min_accuracy,
    )
    # 세 작업에 동일한 설정 적용함. 리스트 순서로 실행 순서를 지정하는 것은 아님
    for task in [prep, fitted, evaluated]:
        # CPU 0.1코어 요청함. 사용량을 0.1코어로 고정하는 설정은 아님
        task.set_cpu_request("100m")
        # CPU 사용 상한을 1코어로 설정함
        task.set_cpu_limit("1")
        # 스케줄링 시 메모리 256Mi 요청함
        task.set_memory_request("256Mi")
        # 패키지 설치를 고려해 메모리 상한 1Gi 설정함
        task.set_memory_limit("1Gi")
        # 수치 계산 라이브러리의 스레드를 제한하여 자원 사용 억제함
        task.set_env_variable("OMP_NUM_THREADS", "1")
        task.set_env_variable("OPENBLAS_NUM_THREADS", "1")
        task.set_env_variable("MKL_NUM_THREADS", "1")
        # 첫 실습에서는 단계별 실제 실행과 학습 시간 관찰을 위해 캐시 해제
        task.set_caching_options(False)


# 파일을 Python으로 직접 실행할 때 YAML 컴파일 수행함
if __name__ == "__main__":
    # 컴포넌트 본문을 실행하지 않고 작업 사양·입출력 연결을 YAML로 변환함
    compiler.Compiler().compile(
        pipeline_func=iris_pipeline, package_path="iris_pipeline.yaml",
    )
    print("iris_pipeline.yaml 생성 완료")
```

## 6. 코드를 읽으며 확인할 사항

### 6.1. preprocess: 데이터 분할과 정규화

- `load_iris()`로 내장 데이터를 읽음
- `stratify`를 사용하여 클래스 비율을 유지하며 분할함
- `StandardScaler.fit_transform()`은 학습 데이터에만 적용함
- 평가 데이터에는 이미 학습한 scaler의 `transform()`만 적용함
- 학습·평가 배열을 각각 Dataset 아티팩트의 `.path`에 저장함
- `Input[Dataset]`는 메모리에 있는 NumPy 배열을 직접 전달하는 방식이 아니라 아티팩트 참조를 전달하는 방식임
- 패키지 설치·실행 코드는 컨테이너에서 동작하므로 각 함수 안에 필요한 import를 작성함

### 6.2. train: 학습과 모델 저장

- `n_estimators`: 트리 수 / `max_depth`: 각 트리의 최대 깊이
- `n_jobs=1`로 내부 병렬 학습 제한함
- 학습 자체의 실행 시간만 `train_seconds`로 기록함
- classifier와 정규화 통계를 하나의 모델 파일에 저장함
- 아티팩트 metadata에 알고리즘과 주요 파라미터 기록함
- joblib 파일은 신뢰할 수 있는 본 실습 산출물만 읽음

### 6.3. evaluate: 메트릭과 품질 기준

- `Metrics.log_metric()`으로 숫자 지표 기록함
- `ClassificationMetrics`로 혼동행렬 기록함
- 모든 지표를 JSON 보고서 아티팩트와 로그에도 출력함
- `quality_pass=1`: 품질 기준 통과 / `0`: 기준 미달
- 기준 미달이더라도 평가 자체가 정상 완료되면 Run 상태는 Succeeded임
- 반환값 bool은 후속 배포 단계를 조건부로 실행할 때 사용할 수 있는 신호이며, 본 실습에서는 실제 배포를 연결하지 않음

### 6.4. pipeline: 연결과 실행 설정

- preprocess의 학습 Dataset → train의 입력
- preprocess의 평가 Dataset → evaluate의 입력
- train의 Model → evaluate의 입력
- 입력·출력 연결로 실행 순서가 결정되어 `.after()` 추가 불필요
- preprocess 다음에 train, train 다음에 evaluate가 실행됨
- 컴파일 시 함수들이 로컬에서 학습을 수행하는 것이 아니라 컨테이너 실행 스펙과 DAG가 YAML에 생성됨

## 7. 컴파일 및 업로드

JupyterLab의 Terminal에서 다음 명령 실행함:

```bash
cd ~/work/kfp-iris-lab
python iris_pipeline.py
ls -lh iris_pipeline.yaml
```

Notebook 셀에서 실행할 경우 파일이 있는 폴더에서 다음 코드 사용함:

```python
# 현재 커널과 같은 Python으로 파일 실행함
# 노트북의 현재 작업 폴더에 iris_pipeline.py가 있어야 함
import subprocess
import sys

subprocess.run([sys.executable, "iris_pipeline.py"], check=True)
```

예상 메시지:

```text
iris_pipeline.yaml 생성 완료
```

1. Kubeflow 대시보드에서 Pipelines 메뉴 이동함
2. Upload pipeline에서 `iris_pipeline.yaml` 업로드함
3. Pipeline 이름을 `iris-mlops-light`로 지정함
4. Experiments에서 `iris-mlops-lab` Experiment 생성함
5. 해당 Pipeline으로 Create run 수행하고 위 Experiment 선택함
6. Run 유형은 일회성 실행으로 지정함

화면 명칭과 위치는 배포 버전에 따라 차이가 있을 수 있음. 별도 API 인증이나 쿠키 설정을 요구하지 않도록 기본 실습은 대시보드 업로드 방식으로 진행함.

JupyterLab 파일 브라우저에서 `iris_pipeline.yaml`을 우클릭하여 Download 선택함. 내려받은 파일을 Pipelines의 Upload pipeline에서 업로드함. Notebook과 다른 탭에서 대시보드를 열어 작업 가능함.

## 8. Run A·B·C 실행: 한 번에 하나의 조건 변경

모든 Run에서 `test_size=0.3`, `random_state=42`, `min_accuracy=0.9` 고정함. 같은 Pipeline과 같은 Experiment를 사용함.

| Run 이름 | n_estimators | max_depth | 비교 목적 |
|---|---:|---:|---|
| iris-A-depth1 | 10 | 1 | 얕은 트리를 사용한 기준 실험 |
| iris-B-depth2 | 10 | 2 | A와 비교: 깊이만 변경 |
| iris-C-trees30 | 30 | 2 | B와 비교: 트리 수만 변경 |

1. Pipelines에서 업로드한 `iris-mlops-light`와 동일한 버전을 선택함
2. Create run에서 Run 이름 `iris-A-depth1`과 Experiment `iris-mlops-lab` 지정함
3. Run 유형은 일회성으로 설정하고 표의 파라미터 및 고정값 모두 확인함
4. A 실행 후 완료 상태 확인함
5. 같은 Pipeline 버전으로 새 Run을 만들고 이름 `iris-B-depth2` 지정함
6. Experiment는 반드시 기존 `iris-mlops-lab` 선택함. max_depth만 2로 변경함
7. B 종료 후 같은 방법으로 `iris-C-trees30` 생성함. B 대비 n_estimators만 30으로 변경함

각 Run에서 DAG의 preprocess·train·evaluate가 Succeeded인지 확인하고 메트릭 및 평가 로그 관찰함. 실행 중 캐시 설정을 별도로 선택할 수 있으면 기존 결과 재사용이 아닌 실제 실행으로 설정함.

동일 코드에 하이퍼파라미터만 변경하는 경우 YAML을 다시 업로드할 필요 없음. 코드·패키지·자원 설정을 변경한 경우에는 재컴파일하고 새 Pipeline 버전을 업로드함.

## 9. Run별 메트릭 확인 및 비교

### 9.1. 평가 노드에서 확인

Run 상세 화면 → evaluate 노드 선택 → 출력 아티팩트 확인함.

| 출력 | 확인 내용 |
|---|---|
| metrics | accuracy, macro_f1, train_seconds, model_size_kib 등의 숫자 |
| confusion | 클래스별 혼동행렬: 행은 실제 클래스, 열은 예측 클래스 |
| report | 전체 지표·파라미터가 포함된 JSON 파일 |
| Output | 품질 통과 여부 bool |

UI에서 메트릭 카드나 표가 표시되지 않으면 metrics 아티팩트의 metadata를 확인함. evaluate 로그에도 동일한 JSON이 출력되므로 반드시 값을 확인할 수 있음. Metrics의 숫자가 모든 버전에서 자동으로 동일한 그래프 형태로 표시된다고 가정하지 않음.

### 9.2. Experiment에서 Run 비교

1. `iris-mlops-lab` Experiment 열기
2. A·B·C Run 체크박스 선택
3. Compare runs 또는 해당 버전의 비교 기능 선택
4. Parameters에서 실제 입력한 `n_estimators`, `max_depth` 확인
5. Metrics 영역에서 같은 이름의 메트릭을 나란히 비교
6. 비교 화면이 해당 v2 아티팩트의 숫자를 집계하지 못하는 버전에서는 Run 상세 UI에서 evaluate의 metrics 아티팩트 metadata 또는 Logs 탭을 열어 아래 표 작성

| Run | 트리 수 | 깊이 | accuracy | macro_f1 | train_seconds | model_size_kib | quality_pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 10 | 1 | | | | | |
| B | 10 | 2 | | | | | |
| C | 30 | 2 | | | | | |

### 9.3. 지표 해석

| 지표 | 의미 | 해석 |
|---|---|---|
| accuracy | 전체 평가 샘플 중 정답 비율 | 높을수록 좋음 |
| macro_f1 | 클래스별 F1 점수를 동일 비중으로 평균 | 높을수록 좋음 |
| train_seconds | classifier.fit()에 걸린 시간 | 작은 모델에서는 측정 잡음이 커 단일 측정으로 우열 단정 금지 |
| model_size_kib | 저장된 모델 묶음의 크기 | 동일 성능이라면 작은 모델이 저장·전달에 유리 |
| test_samples | 평가 샘플 수 | 비교 Run끼리 동일한지 확인 |
| quality_pass | accuracy 기준 통과 여부 | 실행 성공 여부와는 다른 판단 |

- accuracy와 macro_f1은 0~1 범위임. 0.9333은 약 93.33%를 의미함
- 트리 수나 깊이를 늘려도 정확도가 반드시 올라가지는 않음
- 기본 평가 샘플 45개에서는 정답 1개 차이가 약 2.22%p 차이를 만듦
- 학습 시간에는 이미지 pull·패키지 설치·Pod 대기 시간이 포함되지 않음
- Run 전체 Duration은 플랫폼 처리까지 포함하여 별도로 관찰함
- 실제 모델 선택에서는 validation 또는 교차검증으로 튜닝하고, 최종 test는 선택 완료 후 한 번 평가하는 구조가 바람직함. 이 실습의 고정 평가 분할 반복 비교는 그 원리를 익히기 위한 단순화임

### 9.4. 로컬 검증에서 얻은 참고 결과

아래는 작성 코드의 함수 로직을 지정한 라이브러리 버전으로 로컬 실행하여 확인한 값임. KFP 클러스터 Run에서 측정한 결과는 아님. 환경이 다르면 일부 값이 달라질 수 있으며 학생은 자신의 Run 값을 기록함.

| Run | accuracy | macro_f1 | 모델 묶음 크기(KiB, 약) |
|---|---:|---:|---:|
| A | 0.933333 | 0.932660 | 8.28 |
| B | 0.911111 | 0.910714 | 10.86 |
| C | 0.911111 | 0.910714 | 28.11 |

이 예시에서는 복잡도를 늘려도 성능이 개선되지 않음. 작은 평가 집합의 단일 결과이므로 A가 모든 데이터에서 가장 좋은 모델이라고 일반화하지 않음.

## 10. 선택 실습: 데이터 분할 변경 후 결과 비교

원본 Iris 데이터는 유지하고 `random_state`만 변경하여 학습·평가 샘플 구성을 바꿔 봄. 현재 코드에서는 같은 random_state가 모델 난수에도 사용되므로 결과 변화에는 데이터 분할과 모델 난수의 영향이 함께 포함됨.

| Run 이름 | n_estimators | max_depth | test_size | random_state | min_accuracy |
|---|---:|---:|---:|---:|---:|
| iris-B-depth2 | 10 | 2 | 0.3 | 42 | 0.9 |
| iris-F-seed7 | 10 | 2 | 0.3 | 7 | 0.9 |
| iris-G-seed21 | 10 | 2 | 0.3 | 21 | 0.9 |

- B는 기존 Run 재사용함. F·G만 추가로 순차 실행함
- 모든 Run을 `iris-mlops-lab` Experiment에 생성함
- Experiment에서 B·F·G를 선택하여 Parameters 및 Metrics 비교함
- 같은 설정의 모델도 데이터 분할·난수에 따라 성능이 달라질 수 있음을 확인함
- 모델 간 공정한 비교 시 각 모델에 동일한 시드 목록을 적용함
- test_size 변경도 가능하지만 평가 샘플 수가 달라지므로 첫 하이퍼파라미터 비교에서는 고정함
- 원본 데이터셋 자체를 변경하려면 preprocess 코드 수정과 재컴파일 필요함

| Run | random_state | test_samples | accuracy | macro_f1 | 관찰 내용 |
|---|---:|---:|---:|---:|---|
| B | 42 | 45 | | | |
| F | 7 | 45 | | | |
| G | 21 | 45 | | | |

## 11. 확장 실습: 재현성·품질 기준·캐시

### 11.1. Run D: 동일 조건 재실행

- Run 이름 `iris-D-repeat-B`
- B와 동일: `n_estimators=10`, `max_depth=2`, `test_size=0.3`, `random_state=42`, `min_accuracy=0.9`
- B와 D의 accuracy·macro_f1 비교함
- 같은 데이터·코드·라이브러리·난수 시드에서는 같은 성능을 기대함
- train_seconds는 실행 부하·시간 측정에 따라 달라지는 값이므로 동일할 필요 없음
- random_state 고정만으로 모든 하드웨어·라이브러리 환경에서 완전한 재현성을 보장하는 것은 아님

### 11.2. Run E: 품질 기준 미달

- Run 이름 `iris-E-quality-check`
- B와 동일한 조건에서 `min_accuracy=0.99`만 변경함
- 기본 검증 결과처럼 accuracy가 0.99 미만이면 quality_pass=0 확인함
- Run은 Succeeded이지만 모델은 정한 기준을 충족하지 못함
- 후속 운영 설계에서는 bool 반환값을 `dsl.If` 조건으로 연결하여 기준 통과 모델만 배포하는 구조로 확장 가능함
- accuracy와 quality_pass의 차이를 설명함

### 11.3. 선택: 캐시로 반복 비용 줄이기

기본 코드에서는 실제 단계 실행을 관찰하기 위해 캐시를 해제함.

```python
task.set_caching_options(False)
```

이 줄을 아래처럼 수정 후 다시 컴파일·업로드함:

```python
task.set_caching_options(True)
```

같은 입력·컴포넌트 사양으로 다시 실행하면 재사용 가능한 단계의 결과가 캐시에서 반환될 수 있음. UI의 캐시 설정이 코드 설정을 덮어쓰는 환경에서는 실행 설정도 확인함.

- n_estimators만 바꾸면 동일 전처리 결과 재사용 가능
- 실제 캐시 사용 여부는 백엔드·실행 설정·기존 캐시 존재 여부에 따라 확인함
- 캐시된 train 결과에는 과거 학습의 train_seconds가 들어 있으므로 현재 Run에서 다시 학습한 시간으로 해석하면 안 됨
- 캐시 결과 재사용은 새로운 학습 실행의 재현성 확인과 구분함

## 12. 자원 관찰과 문제 해결

### 12.1. 진행 상태 관찰

```bash
kubectl get pods -A -w
```

종료: Ctrl+C.

작업 Pod가 위치한 사용자 namespace와 Pod 이름을 확인한 뒤 실행함:

```bash
kubectl describe pod POD_NAME -n USER_NAMESPACE
kubectl logs POD_NAME -n USER_NAMESPACE --all-containers=true --tail=100
```

metrics-server가 설치된 경우 자원 확인함:

```bash
kubectl top pods -n USER_NAMESPACE
```

| 증상 | 확인 및 조치 |
|---|---|
| Pending | describe의 Events에서 CPU·메모리 부족, PVC 및 스케줄링 문제 확인 |
| ImagePullBackOff | Docker Hub 접근·이미지 주소·레지스트리 제한 확인 |
| pip 설치 실패 | 작업 Pod의 PyPI 연결·DNS·프록시·디스크 용량 확인 |
| No module named sklearn | packages_to_install 포함 여부 확인하고 YAML 재생성 |
| OOMKilled | 설치 단계인지 학습 단계인지 확인. 필요 시 해당 작업 메모리 상한만 2Gi로 변경 |
| Dataset 파일 없음 | .path에 정확히 저장했는지 확인. NumPy 확장자 자동 추가를 피하는 본문 코드 사용 |
| 401·403·로그인 화면 | 올바른 로그인과 사용자 namespace 접근 권한 확인 |
| Run 성공, 메트릭 안 보임 | UI에서 evaluate의 metrics 아티팩트 metadata 및 Logs 탭 확인 |
| accuracy 값이 동일함 | 정상일 수 있음. 혼동행렬·모델 크기도 함께 비교 |
| 학습이 너무 느림 | fit 시간과 전체 Duration 구분. 보통 패키지 설치·Pod 시작 비용 확인 |

실습 전체 코드는 Dataset·Model 등 타입을 사용한 v2 IR YAML용임. legacy v1 예제의 `ContainerOp`, `mlpipeline-metrics` 파일 방식과 섞지 않음.

## 13. 수업 운영 시 설치 비용 줄이기

기본 예제는 Docker 빌드 없이 시작하도록 lightweight Python component를 사용함. 다만 각 단계에서 패키지 설치가 반복되므로 다수 학생이 함께 실행하면 네트워크·설치 비용이 생김.

교수자 사전 준비 시 다음 패키지를 포함한 공통 이미지를 한 번 빌드하여 배포 가능함:

```dockerfile
FROM python:3.10-slim
RUN python -m pip install --no-cache-dir \
    kfp==2.5.0 numpy==1.26.4 scipy==1.13.1 \
    scikit-learn==1.5.2 joblib==1.4.2 threadpoolctl==3.5.0
```

세 컴포넌트의 데코레이터를 다음 구조로 변경함:

```python
@dsl.component(
    base_image="YOUR_REGISTRY/kfp-iris:lab-v1",
    install_kfp_package=False,
)
def preprocess():  # 실제 적용 시 기존 함수의 인자와 본문 유지
    pass
```

- `YOUR_REGISTRY`는 클러스터가 접근할 수 있는 실제 레지스트리로 변경함
- packages_to_install을 제거하고 공통 이미지 안의 패키지 사용함
- 이미지 빌드·업로드 후 YAML 재컴파일·업로드함
- 단계별 패키지 설치를 없애고 노드 이미지 캐시를 활용할 수 있음
- 초기 이미지 다운로드는 여전히 필요함
- 장기 운영에서는 이미지 태그뿐 아니라 digest까지 고정하는 방법 검토함
- 기본 학습 예제를 완료하기 위해 이 절차가 필수인 것은 아님

## 14. 제출 및 점검

제출물:

1. `iris_pipeline.py`, `iris_pipeline.yaml`
2. A·B·C Run의 완료 화면 또는 실행 식별 정보
3. Experiment의 Run 비교 화면 캡처와 작성한 Run 비교표
4. 다음 질문에 대한 짧은 답변

점검 질문:

- 파라미터와 아티팩트의 차이는 무엇이며, 이 코드의 예시는 무엇인가?
- 각 단계가 다른 컨테이너인데 Dataset과 Model을 어떻게 주고받는가?
- 전체 데이터를 먼저 정규화한 뒤 분할하면 어떤 문제가 생기는가?
- 트리 수를 늘렸는데 accuracy가 같다면 무엇을 추가 비교할 수 있는가?
- Succeeded와 quality_pass=1은 어떻게 다른가?
- 이 실습에서 구현한 MLOps 요소와 아직 구현하지 않은 운영 요소는 무엇인가?

교수자 확인 기준:

- 단계별 코드와 입출력 연결이 올바르게 정의됨
- 동일 분할에서 한 번에 하나의 하이퍼파라미터를 변경함
- accuracy뿐 아니라 F1·학습 시간·모델 크기를 비교함
- 성능 차이가 없거나 성능이 감소해도 그 이유와 한계를 해석함
- 재현성·데이터 누수·품질 기준·캐시 개념을 구분함

## 15. 정리 및 참고

- 실습 완료 후 진행 중인 Run이 없는지 확인함
- 불필요한 Notebook은 종료함
- 결과 기록 전 Experiment나 아티팩트 저장소를 삭제하지 않음
- Run 삭제 또는 보관만으로 모든 저장소 파일이 자동 삭제된다고 가정하지 않음
- 공용 클러스터의 PVC·namespace·플랫폼 서비스는 학생이 임의 삭제하지 않음

공식 참고 문서:

- [KFP 파이프라인 개요](https://www.kubeflow.org/docs/components/pipelines/overview/)
- [Lightweight Python Components](https://www.kubeflow.org/docs/components/pipelines/user-guides/components/lightweight-python-components/)
- [아티팩트 생성·전달·추적](https://www.kubeflow.org/docs/components/pipelines/user-guides/data-handling/artifacts/)
- [KFP UI와 실행 비교](https://www.kubeflow.org/docs/components/pipelines/interfaces/)
- [KFP SDK DSL API](https://kubeflow-pipelines.readthedocs.io/en/latest/source/dsl.html)
- [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [데이터 누수와 전처리 주의사항](https://scikit-learn.org/stable/common_pitfalls.html)

검증 범위: 기존 예제는 사용자 환경에서 파이프라인 실행 성공 확인됨. 개정 코드의 Python 구문을 검증함. 함수 로직은 기존 컴파일·실행 성공 예제를 유지하고 설명 주석을 추가함. 이번 개정의 재컴파일은 검증 환경의 패키지 설치 제한으로 수행하지 못함. 개정본을 사용자 클러스터에 직접 제출한 검증은 수행하지 않음.
