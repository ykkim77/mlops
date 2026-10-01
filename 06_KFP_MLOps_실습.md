# 6주차 실습: KFP로 만드는 MLOps 파이프라인

> 「AI플랫폼 06. MLOps 파이프라인 구축 실습」 강의자료 
> 대상: KFP v2 백엔드가 설치된 Kubeflow 환경 / GPU 불필요
> 문서의 명령은 Ubuntu 또는 Windows Terminal의 WSL Ubuntu에서 실행함

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
| 핵심 실행 | A·B·C 3개 Run을 순차 실행 |
| 확장 실행 | D: 재현성 확인 / E: 품질 기준 미달 확인 |
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

### 4.2. Python 가상환경

본 예제의 검증 조합은 Python 3.10 계열 컨테이너와 KFP SDK 2.5.0임. 로컬 작성 환경은 Python 3.9~3.11 권장. SDK 2.5.0은 예제의 재현성을 위한 고정 버전이며 최신 버전이라는 의미가 아님. 실제 백엔드 버전과 운영 정책에 따라 조정 시 재컴파일·검증 필요.

```bash
python3 --version
mkdir -p ~/kfp-iris-lab
cd ~/kfp-iris-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install kfp==2.5.0
python -c "import kfp; print(kfp.__version__)"
```

`venv`가 없다는 오류가 발생하면 Ubuntu 환경에서 설치 후 다시 실행함:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv
```

- 로컬 PC는 코드 작성과 YAML 컴파일을 담당함
- 학습 패키지는 아래 `packages_to_install`에 따라 각 작업 컨테이너에 설치됨
- 로컬에서 scikit-learn을 설치하는 것만으로 클러스터의 작업 컨테이너에 설치되지는 않음
- IDE에서 새 파일 `iris_pipeline.py`를 생성하고 다음 전체 코드를 복사함
- Ubuntu 터미널 편집 시 `nano iris_pipeline.py` 사용 가능. 저장은 Ctrl+O → Enter, 종료는 Ctrl+X

## 5. 전체 파이프라인 코드

파일명: `iris_pipeline.py`

```python
from kfp import compiler, dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics, ClassificationMetrics

# 컴포넌트 내부에서 사용하는 패키지는 컨테이너에도 설치되어야 함
PACKAGES = [
    "numpy==1.26.4", "scipy==1.13.1", "scikit-learn==1.5.2",
    "joblib==1.4.2", "threadpoolctl==3.5.0",
]


@dsl.component(base_image="python:3.10-slim", packages_to_install=PACKAGES)
def preprocess(
    test_size: float,
    random_state: int,
    train_data: Output[Dataset],
    test_data: Output[Dataset],
):
    import os
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size는 0.1~0.5 범위로 지정")
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=test_size,
        random_state=random_state, stratify=iris.target,
    )
    # 반드시 분할 후 학습 데이터에서만 평균/표준편차 학습: 데이터 누수 방지
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    for artifact, x, y in [(train_data, x_train, y_train), (test_data, x_test, y_test)]:
        os.makedirs(os.path.dirname(artifact.path), exist_ok=True)
        # 파일 객체 사용: np.savez가 .npz 확장자를 자동 추가하는 문제 방지
        with open(artifact.path, "wb") as f:
            np.savez_compressed(
                f, X=x, y=y, mean=scaler.mean_, scale=scaler.scale_,
                labels=np.array(iris.target_names),
            )
        artifact.metadata.update({
            "dataset": "sklearn-iris", "rows": int(len(y)),
            "features": 4, "random_state": random_state,
            "test_size": test_size, "scaler_fit": "train-only",
        })
    print(f"train={len(y_train)}, test={len(y_test)}")


@dsl.component(base_image="python:3.10-slim", packages_to_install=PACKAGES)
def train(
    train_data: Input[Dataset],
    n_estimators: int,
    max_depth: int,
    random_state: int,
    model: Output[Model],
):
    import os
    import time
    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    if not 1 <= n_estimators <= 100:
        raise ValueError("경량 실습을 위해 n_estimators는 1~100으로 제한")
    if not 1 <= max_depth <= 10:
        raise ValueError("max_depth는 1~10으로 제한")
    with np.load(train_data.path, allow_pickle=False) as data:
        x, y = data["X"], data["y"]
        mean, scale = data["mean"], data["scale"]
        labels = data["labels"].tolist()
    classifier = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=random_state, n_jobs=1,
    )
    start = time.perf_counter()
    classifier.fit(x, y)
    fit_seconds = time.perf_counter() - start
    # 추론에서도 동일 전처리를 재사용할 수 있도록 통계와 모델을 함께 보관
    bundle = {
        "classifier": classifier, "scaler_mean": mean, "scaler_scale": scale,
        "labels": labels, "fit_seconds": fit_seconds,
        "n_estimators": n_estimators, "max_depth": max_depth,
        "random_state": random_state,
    }
    os.makedirs(os.path.dirname(model.path), exist_ok=True)
    joblib.dump(bundle, model.path)
    model.metadata.update({
        "algorithm": "RandomForestClassifier", "n_estimators": n_estimators,
        "max_depth": max_depth, "random_state": random_state,
        "sklearn_version": "1.5.2",
    })
    print(f"fit_seconds={fit_seconds:.6f}")


@dsl.component(base_image="python:3.10-slim", packages_to_install=PACKAGES)
def evaluate(
    model: Input[Model],
    test_data: Input[Dataset],
    min_accuracy: float,
    metrics: Output[Metrics],
    confusion: Output[ClassificationMetrics],
    report: Output[Dataset],
) -> bool:
    import json
    import os
    import numpy as np
    import joblib
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    if not 0.0 <= min_accuracy <= 1.0:
        raise ValueError("min_accuracy는 0~1 범위로 지정")
    bundle = joblib.load(model.path)
    with np.load(test_data.path, allow_pickle=False) as data:
        x, y = data["X"], data["y"]
    # x는 preprocess 단계에서 이미 정규화됨: 다시 정규화하지 않음
    predicted = bundle["classifier"].predict(x)
    accuracy = float(accuracy_score(y, predicted))
    macro_f1 = float(f1_score(y, predicted, average="macro", zero_division=0))
    passed = accuracy >= min_accuracy
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
    for name, value in values.items():
        metrics.log_metric(name, float(value))
    matrix = confusion_matrix(y, predicted, labels=[0, 1, 2])
    confusion.log_confusion_matrix(bundle["labels"], matrix.tolist())
    result = {
        "metrics": values, "min_accuracy": min_accuracy,
        "random_state": bundle["random_state"],
        "confusion_matrix": matrix.tolist(), "labels": bundle["labels"],
    }
    os.makedirs(os.path.dirname(report.path), exist_ok=True)
    with open(report.path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    report.metadata["format"] = "json"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return passed


@dsl.pipeline(name="iris-mlops-light", description="Iris 전처리-학습-평가 및 Run 메트릭 비교")
def iris_pipeline(
    n_estimators: int = 10,
    max_depth: int = 2,
    test_size: float = 0.3,
    random_state: int = 42,
    min_accuracy: float = 0.9,
):
    prep = preprocess(test_size=test_size, random_state=random_state)
    fitted = train(
        train_data=prep.outputs["train_data"], n_estimators=n_estimators,
        max_depth=max_depth, random_state=random_state,
    )
    evaluated = evaluate(
        model=fitted.outputs["model"], test_data=prep.outputs["test_data"],
        min_accuracy=min_accuracy,
    )
    for task in [prep, fitted, evaluated]:
        task.set_cpu_request("100m")
        task.set_cpu_limit("1")
        task.set_memory_request("256Mi")
        task.set_memory_limit("1Gi")
        task.set_env_variable("OMP_NUM_THREADS", "1")
        task.set_env_variable("OPENBLAS_NUM_THREADS", "1")
        task.set_env_variable("MKL_NUM_THREADS", "1")
        # 첫 실습에서는 단계별 실제 실행과 학습 시간 관찰을 위해 캐시 해제
        task.set_caching_options(False)


if __name__ == "__main__":
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

가상환경을 활성화한 실습 폴더에서 실행함:

```bash
python iris_pipeline.py
ls -lh iris_pipeline.yaml
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

WSL에서 Windows 탐색기로 현재 폴더를 열 때:

```bash
explorer.exe .
```

서버에서 컴파일한 경우 로컬 PC 터미널의 파일 전송 예시:

```bash
scp user@SERVER:~/kfp-iris-lab/iris_pipeline.yaml .
```

`user`, `SERVER`는 실제 계정·주소로 변경함.

## 8. Run A·B·C 실행: 한 번에 하나의 조건 변경

모든 Run에서 `test_size=0.3`, `random_state=42`, `min_accuracy=0.9` 고정함. 같은 Pipeline과 같은 Experiment를 사용함.

| Run 이름 | n_estimators | max_depth | 비교 목적 |
|---|---:|---:|---|
| iris-A-depth1 | 10 | 1 | 얕은 트리를 사용한 기준 실험 |
| iris-B-depth2 | 10 | 2 | A와 비교: 깊이만 변경 |
| iris-C-trees30 | 30 | 2 | B와 비교: 트리 수만 변경 |

1. A의 파라미터 입력 후 실행함
2. DAG의 preprocess·train·evaluate가 Succeeded인지 확인함
3. 메트릭과 평가 로그 확인함
4. A 종료 후 B 실행함
5. B 종료 후 C 실행함

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
6. 비교 화면이 해당 v2 아티팩트의 숫자를 집계하지 못하는 버전에서는 Run별 metrics metadata 또는 report JSON을 사용하여 아래 표 작성

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

## 10. UI에 의존하지 않는 CSV 비교 방법

선택 실습이지만, UI의 비교 화면이 제한된 경우에도 Run별 비교를 완성할 수 있는 방법임.

1. 각 Run의 report 아티팩트를 다운로드함
2. 각각 `reports/iris-A-depth1.json`, `reports/iris-B-depth2.json`, `reports/iris-C-trees30.json`으로 저장함
3. 다운로드 기능이 없으면 evaluate 로그의 JSON 객체만 복사하여 같은 파일명으로 저장함. 로그 시간·로그 접두어·기타 출력은 포함하지 않음
4. 실습 폴더에 아래 코드를 `compare_reports.py`로 저장함

```python
import csv
import json
from pathlib import Path

fields = [
    "run", "n_estimators", "max_depth", "accuracy", "macro_f1",
    "train_seconds", "model_size_kib", "test_samples", "quality_pass",
]
files = sorted(Path("reports").glob("*.json"))
if not files:
    raise SystemExit("reports 폴더에 Run별 JSON 보고서를 먼저 저장")
rows = []
for file in files:
    result = json.loads(file.read_text(encoding="utf-8"))
    rows.append({"run": file.stem, **result["metrics"]})
with open("run_comparison.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
for row in rows:
    print(
        f'{row["run"]}: accuracy={row["accuracy"]:.4f}, '
        f'F1={row["macro_f1"]:.4f}, '
        f'fit={row["train_seconds"]:.6f}s, '
        f'size={row["model_size_kib"]:.2f}KiB, '
        f'pass={row["quality_pass"]}'
    )
print("run_comparison.csv 생성 완료")
```

실행함:

```bash
mkdir -p reports
# 위 폴더에 Run별 JSON 저장 후 실행
python compare_reports.py
```

생성된 `run_comparison.csv`를 Excel 등에서 열어 Run별 값 비교 가능. 별도 pandas나 MLflow 설치 불필요.

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
| Run 성공, 메트릭 안 보임 | evaluate의 metrics metadata·report JSON·로그 확인 후 CSV 비교 사용 |
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
3. Run 비교표 또는 `run_comparison.csv`
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

검증 범위: KFP SDK 2.5.0으로 YAML 컴파일 성공, 전처리·학습·평가 함수 로직의 로컬 실행 및 A·B·C·D 성능 값 확인. 실제 사용자의 Kubeflow 클러스터에 제출·실행한 검증은 수행하지 않음.
