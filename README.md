# 뉴스링고 (Newslingo)

> 뉴스 기사 기반 개인화 영어 학습 Agent — 6반 5조 AI Agent 설계서 구현체
> LangChain 1.x (`create_agent` + middleware + Structured Output) · Streamlit UI

사용자가 원하는 주제의 **영문 뉴스**(Guardian Open Platform)를 추천받아
원문·번역·어휘·문법으로 학습하고, 기사 기반 **자유 대화**와 **퀴즈**로 실력을 점검한다.
난이도는 퀴즈 성과에 따라 **제안만** 되고, 사용자가 **직접 승인(HITL)** 해야만 바뀐다.

---

## 1. 설치 & 실행

```bash
cd 0909-0911_langchain/newslingo
pip install -r requirements.txt

cp .env.example .env   # 그리고 키 채우기
#  - OPENAI_API_KEY 필수
#  - GUARDIAN_API_KEY 없으면 .env 에서 NEWSLINGO_USE_MOCK_NEWS=true (샘플 기사로 동작)

# 메인 데모 — Streamlit UI
streamlit run app.py

# 터미널 데모 (설계서 2.2 플로우를 콘솔에서 손으로 따라가 보기)
python -m newslingo.app_cli
```

`OPENAI_API_KEY` 가 없으면 `app.py` 는 자동으로 `mock_service.py`(고정 샘플 응답)로
전환되어 API 키 없이도 UI 확인이 가능하다.

---

## 2. 파일 구조 — 담당자 / 설계서 섹션 매핑

역할군을 기준으로 3명이 파일을 나눠 담당한다. 같은 역할군 안에서는 서로의 파일을
직접 고치지 않고, 인터페이스(함수 시그니처)만 맞춰 조율한다.

| 역할군 | 담당 | 파일 | 설계서 |
|---|---|---|---|
| **AgentCore** | 박서윤 | `agent.py` | **2.1** Agent 조립 |
| | | `middleware.py` | **3.2** Middleware |
| | | `prompts/main.py` | 2.3 System Prompt |
| | | `memory.py` | **3.1** 단기/장기 메모리 |
| | | `service.py` | **2.2** 오케스트레이션 (`LearningSession`) |
| | | `config.py` | 1.5, 2.3 환경변수·상수 |
| | | `app_cli.py`, `app.py`, `mock_service.py` | — 진입점(터미널/Streamlit/목업) |
| **Structured Output & 로직** | 이지영 | `schemas.py` | **2.4** Pydantic 스키마 |
| | | `chains.py` | 2.4 LCEL 체인 (추천/학습자료/퀴즈) |
| | | `prompts/generation.py` | 2.4 체인 프롬프트 템플릿 |
| | | `grading.py` | 2.2 (10~11) 채점·난이도 추천 |
| **Tools & 가드레일** | 박태식 | `tools.py` | **2.5** `news_search`, `update_preference` |
| | | `guardrails.py` | **3.3** 입력 2단계 가드레일 |
| | | `prompts/guardrail.py` | 3.3 가드레일 System Prompt |
| 공통 | — | `tests/` | 4. 테스트 설계 |
| | | `notebooks/` | 발표용 워크스루 |
| | | `architecture-diagram.svg` | 전체 구조도 |

---

## 3. 내부 흐름 — Agent가 하는 일 vs 체인이 하는 일

가장 먼저 이해해야 할 설계 원칙: **자유 대화만 Agent가 맡고, "정해진 스키마로
정확히 N개를 뽑아야 하는" 생성(기사 추천/학습자료/퀴즈)은 Agent가 아니라
`chains.py`의 LCEL 체인이 맡는다.** Agent의 자율적 tool-calling은 이런 결정적
생성에는 재현성이 떨어지기 때문이다.

```
사용자 접속 → 난이도 선택(버튼) → Store 저장
    ↓
학습 주제 입력
    ↓
┌─ 2단계 Guardrail (guardrails.py) ──────────────────┐
│ 1차: 정규식 금칙어 필터 (모델 호출 없음, 즉시 차단) │
│ 2차: 모델2(gpt-4o-mini, temp=0) + TopicGuardrailResult │
└─────────────────────────────────────────────────────┘
    ↓ 허용
news_search Tool (tools.py, Guardian Open Platform)
    ↓
article_recommender_chain (chains.py) → ArticleCandidateList(5개, Structured Output)
    ↓
사용자 기사 선택(카드 클릭)
    ↓
study_material_chain (chains.py) → ArticleStudyMaterial (원문/번역/용어/문법)
    ↓
┌─ 여기서부터 메인 Agent(agent.py) 가 담당 ───────────────┐
│ 채팅: 기사 원문 + 학습자료를 매 호출마다 system prompt에  │
│       주입(middleware.profile_injection) → 그 근거로만 답변 │
│ Tool: update_preference (선호/난이도 변경, HITL 2차 승인) │
└───────────────────────────────────────────────────────┘
    ↓ (버튼) 퀴즈 응시하기 — 채팅과 독립적으로 언제든 클릭 가능
quiz_chain (chains.py) → QuizSet(영어 5문항, Structured Output)
    ↓
grade_quiz / recommend_level (grading.py, 순수 파이썬 — LLM 미사용)
    ↓
난이도 조정: 추천이 '유지'여도 사용자가 상향/하향/유지 중 직접 선택
    → 선택이 현재값과 다르면 update_preference 호출 → HITL 2차 승인
```

전체 다이어그램: [`architecture-diagram.svg`](./architecture-diagram.svg)

---

## 4. Agent 설계 핵심 — 왜 이렇게 구성했나

### 4.1 Middleware 4종 (`middleware.py`, 설계서 3.2)

| 이름 | Hook | Built-in/Custom | 역할 |
|---|---|---|---|
| `profile_injection` | `wrap_model_call`(`@dynamic_prompt`) | Custom | Store 프로필 + 현재 기사/학습자료 → System Prompt 동적 주입 |
| `HumanInTheLoopMiddleware` | `wrap_tool_call` | Built-in | `update_preference` 실행 전 2차 승인 (approve/edit/reject) |
| `SummarizationMiddleware` | `before_model` | Built-in | 대화 이력이 길어지면 자동 요약 |
| `ModelFallbackMiddleware` | `wrap_model_call` | Built-in | 메인 모델 실패 시 대체 모델로 전환 |

**왜 `@dynamic_prompt`(`wrap_model_call` 축약형)를 썼고, `before_agent`는 왜 안 썼나?**
`before_agent`/`after_agent`는 `agent.invoke()` 한 번당 딱 1번만 실행되는 세션 단위
훅이다. 그런데 한 번의 `invoke()` 안에서도 tool 호출 → 재추론처럼 모델이 여러 번
불릴 수 있고, 그때마다 최신 프로필/기사 컨텍스트가 반영돼야 한다(예: 채팅 중간에
`update_preference`로 level이 바뀌면 바로 다음 모델 호출부터 새 난이도가 적용돼야
함). `before_agent`로는 이 타이밍을 못 맞춰서 제외했고, "매 모델 호출 직전 system
prompt 문자열만 새로 만든다"는 목적엔 `dynamic_prompt`가 가장 적은 코드로 의도가
드러나는 선택이었다.

**Context 설계 (`agent.py`)** — `user_id`뿐 아니라 `article_title`/`article_text`/
`study_material_summary`를 `Context`에 실어 매 `invoke()`마다 새로 채워 넘긴다.
처음엔 프로필(topic/level)만 주입했는데, 그러면 채팅 Agent가 방금 선택한 기사의
원문을 전혀 몰라 엉뚱한 문장을 지어내는 문제가 있었다 — 그래서 세션이 들고 있는
기사/학습자료 상태를 매번 Context로 흘려보내도록 고쳤다.

### 4.2 규칙 기반으로 LLM 판단을 대신하는 지점들

Tool-calling(Agent의 자율 판단)에 전적으로 맡기면 오작동이 잦았던 지점 두 곳은
**LLM에 맡기지 않고 규칙 기반으로 앞단에서 처리**한다.

- **"다른 기사 보여줘" / "그만할래" 같은 제어 의도** — Agent가 `news_search`를
  반복 호출하다 recursion limit 크래시가 나거나, `update_preference`로 잘못
  해석해 엉뚱한 HITL이 뜨는 문제가 실제로 있었다. `service.py`의
  `detect_control_intent()`가 채팅 텍스트를 Agent에게 보내기 전에 키워드로
  먼저 걸러서 "버튼을 눌러주세요"로 안내하고, 실제 전환은 UI 버튼으로만 한다.
- **입력 가드레일 1차 필터** — 금칙어는 모델 호출 없이 정규식으로 즉시 차단하고,
  통과분만 모델2로 넘긴다 (비용·지연 최적화 + 명백한 케이스의 오탐 방지).

### 4.3 HITL 2단계 승인 — 한 메커니즘, 두 곳에서 재사용

"UI 버튼 클릭(1차 의사표시) → `HumanInTheLoopMiddleware` interrupt(2차 재확인)"
패턴을 **채팅 중 선호 변경**과 **퀴즈 후 난이도 조정** 두 군데에서 동일하게 쓴다.
퀴즈 후 난이도 조정은 추천 방향이 '유지'여도 사용자가 상향/하향을 직접 고를 수
있게 했다 — 시스템 추천은 참고만 될 뿐, 최종 결정권은 항상 사용자에게 있다는
설계 원칙을 지키기 위함.

### 4.4 Structured Output — 왜 Agent가 아니라 체인인가

기사 추천/학습자료/퀴즈는 "정확히 N개", "특정 필드 필수" 같은 제약이 있는
생성이라 `chains.py`의 `PromptTemplate | model.with_structured_output(schema)`
파이프라인으로 고정한다. Agent의 자유로운 tool-calling 루프에 맡기면 이런
제약을 매번 정확히 지킨다는 보장이 없다 (실제로 퀴즈 5문항이 3개만 나오는
문제를 겪었고, 개수 검증/재시도 로직이 없다는 게 현재 알려진 한계).

---

## 5. 알려진 한계 / TODO

- `chains.py`의 구조화 출력은 Pydantic 필드 제약(`min_length` 등)에 의존하는데,
  OpenAI 구조화 출력이 배열 길이 제약을 강제하지 않아 개수가 틀어질 수 있음 →
  재시도 로직 필요 (이지영 담당).
- `NEWS_RAW_PAGE_SIZE`만큼 기사를 받아 난이도에 맞는 걸 고르라고 프롬프트로만
  지시 중 — 실제 리딩 난이도(문장 길이/어휘 수준)를 기사 본문에서 측정해
  반영하는 로직은 없음.
- Streamlit UI(`app.py`)는 카드 클릭으로 언제든 기사를 바꿀 수 있어, CLI(`app_cli.py`)
  처럼 "기사를 떠나기 전 퀴즈 필수"가 강제되지 않음 — 의도적 차이(자유 탐색 우선)
  인지 통일할지는 추후 결정.

---

## 6. 테스트

```bash
cd 0909-0911_langchain
python -m pytest newslingo/tests -q
```

LLM 호출 없이 검증 가능한 로직만 다룬다 (가드레일 규칙 필터, 채점, 난이도 추천
임계치). Agent/체인이 실제로 잘 도는지는 `streamlit run app.py` 또는
`python -m newslingo.app_cli`로 직접 확인한다.
