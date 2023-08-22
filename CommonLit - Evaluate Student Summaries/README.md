## Description

**Goal of the Competition**

이 대회의 목표는 3-12학년의 학생들이 작성한 요약글을 평가하는 것이다. 이 task를 위해 요약문의 clarity, precision, fluency 뿐만 아니라 source text의 디테일과 main idea를 얼마나 잘 표현했는가를 평가해야 한다. 이를 위해 실제 학생들이 작성한 요약문 데이터를 이용하여 모델을 만든다. 이러한 작업을 통해 실제 학생을 가르치는 선생님에게 도움을 줄 수 있고 학생들에게 즉각적인 피드백을 제공하는 교육 플랫폼에게도 도움을 줄 수 있다.

****Context****

내용을 요약하는 것은 모든 나이를 불문하고 학습에 있어 중요한 능력이다. 요약 능력은 특히 제2외국어를 학습할 때나 학습 장애가 있는 학생들 사이에서 독해력 향상에 중요한 역할을 한다. 또한, 요약은 비판적 사고를 촉진시키고 글쓰기 능력을 향상시키는 가장 효과적인 방법 중 하나이다. 하지만 요약문을 평가하는 것은 교사에게 굉장히 시간을 많이 요하는 작업이므로 학생들을 요약에 대해서 많은 피드백을 받지 못하는 것이 사실이다. 해당 문제를 해결하는데 있어서 Large Language Model(LLM)을 이용하여 요약문을 신속하게 평가하는데 도움을 줄 수 있다. 

논리적인 글쓰기나 서술형 문장을 작성하는 과정에서 자동화된 평가는 많은 발전이 있어 왔다. 하지만 현존하는 기술들은 요약문을 번역하는데 아직은 크게 효과적이지 않은 부분이 있다. 요약문을 평가하기 위해서는 모델이 학생이 작성한 하나의 글과 긴 문장들에 대한 복잡성을 모두 소화할 수 있어야 한다. 요약문을 평가하기 위한 몇몇 방법들이 연구되어 왔지만 데이터의 부족때문에 실제 학생이 작성했던 글을 평가하기 보다는 자동으로 생성된 요약문을 평가하는데 그치고 있었다. 

해당 대회를 주최한 [CommonLit](https://www.commonlit.org/en)는 비영리 교육 기술 조직이다. CommonLit는 모든 학생들, 특히 그 중에서도 Title 1 학교의 학생들이 대학이나 혹은 그 이상에서 성공하기 위해 필요한 읽기, 쓰기, 의사소통 및 문제 해결 능력을 갖춘 상태에서 졸업을 하기 위해 노력하고 있다. 해당 미션에는 CommonLit 뿐만 아니라 [The Learning Agency Lab](https://www.the-learning-agency-lab.com/), [Vanderbilt University](https://www.vanderbilt.edu/), and [Georgia State University](https://www.gsu.edu/)가 함께 한다. 

해당 과제를 위한 알고리즘을 개발함으로서 이번 대회에 참여하는 모든 사람들이 교사와 학생 모두가 기본적인 글쓰기 능력을 향상시키는데 도움을 줄 수 있다. 이를 통해 학생들은 독해력, 비판적 사고력, 글쓰기 능력을 동시에 향상시키면서 요약 연습을 할 수 있는 기회를 더 많이 갖게 될 것이다. 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c88846f4-2458-4db9-a3d8-176d132b311a/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f240eb04-627f-4a71-9177-6407da0e8e0d/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/46689db1-60fb-4a84-98ee-203011c79046/Untitled.png)

## Evaluation

**Metric**

이번 대회의 평가 지표는 Mean Columnwise Root Mean Squarred Error(MCRMSE)을 이용한다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fb6d2d67-3d32-4f96-9b69-a3f01b8bc96b/Untitled.png)

- $N_t$: number of scored ground truth target columns

**Submission File**

test set의 각각의 student_id에 대해서 두 개의 분석 지표를 예측해야 한다. 형식은 아래와 같다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0eeacd1d-e8ff-4b75-84f3-46be9b460813/Untitled.png)

## Efficiency Prize Evaluation

****Efficiency Prize****

해당 대회에서는 효율성을 위한 평가를 별도로 진행한다. 해당 모델은 real-world 상황에서도 충분히 이용할 수 있어야 하기 때문에 정확도 상승만을 위한 과도하게 무거운 모델은 현실성이 많이 떨어지게 된다. Efficiency Prize를 위해서 아래와 같이 run time과 predictive performance를 동시에 평가한다. 

- Leader board에서 팀이 직접 선택한 submission이나 My Submissions Tab에서 자동으로 선택된 특정 submission이 평가된다.
- sample_submission.csv 보다 더 높은 점수를 기록해야 한다.
- 평가에는 오직 CPU만을 이용한다.

 ****Evaluation Metric****

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/87faf37b-077a-44ee-b17c-dda9abdf43a7/Untitled.png)

- MCRMSE: 기본 성능 지표
- Base: baseline 성능 지표
- min MCRMSE: private leaderboard의 가장 낮은 점수
- RuntimeSeconds: submission이 평가되는데 걸리는 시간

효율성 지표에 대한 점수는 실시간으로 leaderboard에 표시되지 않고 순위만 표시되며 대회가 끝나고 업데이트 된다.
