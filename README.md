# Prompt?

## 종합평가 markdown
```python
def words_for_person(name,
                     height,
                     waist,
                     al_score,
                     act_score,
                     hypertension_proba,
                     hypertension_rank,
                     physical_activity_rank,
                     waist_min,
                     waist_max,
                     waist_min2,
                     waist_max2):
    max_na = 2400
    words = f'''
    {name}님의 설문에 대한 종합 평가는 다음과 같습니다.
    '''
    if hypertension_proba != None:
        words += f'\n- 모델의 결과로는 **{hypertension_proba}%** 의 확률로 고혈압 위험이 있으며,'
    if hypertension_rank != None:
        words += f' {name}님과 비슷한 사람들의 군집은 **{len(filtered_df)}** 명입니다. '
        if hypertension_rank < 10:
            words += f' 이 중에서 고혈압 확률은 상위 **{hypertension_rank}%** 입니다. 고혈압에 필히 주의가 필요합니다.'
        elif hypertension_rank < 30:
            words += f' 이 중에서 고혈압 확률은 상위 **{hypertension_rank}%** 입니다. 고혈압에 주의가 필요할수도 있어요.'
        elif hypertension_rank < 50:
            words += f' 이 중에서 고혈압 확률은 상위 **{hypertension_rank}%** 입니다. 고혈압에 주의가 필요해요.'
        elif hypertension_rank < 70:
            words += f' 이 중에서 고혈압 확률은 상위 **{hypertension_rank}%** 입니다. 나름 건강하신 상태 입니다.'
        else:
            words += f' 이 중에서 고혈압 확률은 상위 **{hypertension_rank}%** 입니다. 매우 건강하신 상태입니다.'
    
    if al_score != None:
        words += f'\n- 음주 점수는 **{al_score}** 점으로,'
        if al_score > 0 and al_score <= 10 :
            words += f' 음주 점수가 낮은편 입니다.'
        elif al_score > 10 and al_score <= 30:
            words += f' 음주 점수가 다소 낮은편 입니다.'
        elif al_score > 30 and al_score <= 70:
            words += f' 음주 점수가 살짝 높습니다.'
        elif al_score > 70 and al_score <= 100:
            words += f' 음주 점수가 매우 높아요.'
        else:
            words += f' 음주를 꼭 줄이셔야 합니다.'
    
    
    if act_score != None:
        words += f'\n- 신체활동 점수는 **{act_score}** 점으로, **{len(filtered_df)}명** 의 사람들 중에서 상위 **{physical_activity_rank}%** 입니다.'
        if act_score >= -1 and act_score <= 3:
            words += f' 신체활동 점수가 낮은편으로 신체활동량을 늘리는 것을 추천합니다.'
        elif act_score > 3 and act_score <= 7:
            words += f' 신체활동 점수가 평범한 편입니다.'
        elif act_score > 7 and act_score <= 8:
            words += f' 신체활동 점수가 높은 편으로 신체활동을 꾸준히 하시고 있습니다.'
        elif act_score > 8 and act_score <= 9:
            words += f' 신체활동 점수가 매우 높습니다. 신체활동을 꾸준히 하시고 있습니다.'
        elif act_score > 9:
            words += f' 혹시 운동 선수 이신가요?'
                
    if height != None:
        words += f'''\n- {name}님의 신장인 **{height} cm** 에서 이상적인 허리둘레는 **{waist_min} cm** ~ **{waist_max} cm** 입니다.
        인치로는 **{waist_min2} 인치** ~ **{waist_max2} 인치** 입니다.
        '''
        if waist_min <= waist <= waist_max:
            words += f' 허리둘레가 이상적인 범위 안에 있어요.'
        if waist < waist_min:
            words += f' 허리둘레가 약 **{round((((waist_min+waist_max)/2) - waist),2)} cm** 벗어났습니다.(약 **{round(((waist_min+waist_max)/2 - waist)/2.54,2)}** 인치)'
        if waist > waist_max:
            words += f' 허리둘레가 약 **{round((waist - ((waist_max+waist_min)/2)),2)} cm** 줄여야합니다.(약 **{round((waist - ((waist_max+waist_min)/2))/2.54,2)}** 인치)'
    if max_na != None:
        if hypertension_proba > 0 and hypertension_proba < 20:
            words += f'\n - {name}님의 권장 한끼 나트륨 권장량은 **{(max_na*0.9)/3}mg** 입니다.'
        if hypertension_proba > 20 and hypertension_proba < 30:
            words += f'\n - {name}님의 권장 한끼 나트륨 권장량은 **{(max_na*0.8)/3}mg** 입니다.'
        if hypertension_proba > 30 and hypertension_proba < 40:
            words += f'\n - {name}님의 권장 한끼 나트륨 권장량은 **{(max_na*0.7)/3}mg** 입니다.'
        if hypertension_proba > 40 and hypertension_proba < 50:
            words += f'\n - {name}님의 권장 한끼 나트륨 권장량은 **{(max_na*0.6)/3}mg** 입니다.'
        if hypertension_proba > 50:
            words += f'\n - {name}님의 권장 한끼 나트륨 권장량은 **{(max_na*0.5)/3}mg** 입니다.'
```
- 다양한 if문을 통해 설문 결과에 따라서 종합평가 markdown을 생성합니다.



## userinfo, name 정의
```python
name = profile_pdf['신체정보']['이름']
    user_info = f"""
    이름: {profile_pdf['신체정보']['이름']}
    나이: {profile_pdf['신체정보']['만 나이']}세
    성별: {profile_pdf['신체정보']['성별']}
    키: {profile_pdf['신체정보']['키']}cm
    체중: {profile_pdf['신체정보']['체중']}kg
    허리둘레: {profile_pdf['신체정보']['허리둘레']}cm
    당뇨병 여부: {profile_pdf['질병 정보']['당뇨병 여부']}
    이상지질혈증 여부: {profile_pdf['질병 정보']['이상지질혈증 여부']}\n
    <{name}님의 종합 평가> \n {words_for_person(profile_pdf['신체정보']['이름'],
                        float(profile_pdf['신체정보']['키']),
                        float(profile_pdf['신체정보']['허리둘레']),
                        int(profile_score['음주 점수']),
                        round(float(profile_score['신체활동 점수']),2),
                        float(profile_score['고혈압 확률']),
                        round(hypertension_rank,2),
                        round(physical_activity_rank,2),
                        round(ideal_waist['waist_min'],2),
                        round(ideal_waist['waist_max'],2),
                        round(ideal_waist2['waist_min'],2),
                        round(ideal_waist2['waist_max'],2)
                        )}
```
- 프롬포트에 들어갈 markdown 변수를 생성합니다


## Prompt


```python
 custom_prompt_template = """
    사용자의 건강정보: {user_info}\n
    요리 레시피: 요리 레시피는 무조건 벡터 데이터베이스에서만 찾아서 제공하며, 벡터 데이터베이스에 없는 요리 레시피는 제공하지 않는다.\n
    목적: {name}의 하루 권장 나트륨 크게 넘지 않는 한끼 정보를 제공한다 \n 
    형식: 해당 레시피에 대한 근거는 건강보고서의 내용을 바탕으로 왜 이음식을 선택했는지 설명한다.\n
    컨텍스트: {context}\n
    질문: {question}\n
    """
```
0. lmstudio에 입력될 custom_prompt는 python print 형식과 markdown 형식 모두 지원하므로 둘의 문법 ('\n' 또는 ** **예시** **) 을 혼용해서 사용해도 큰 문제는 없습니다. 

1. 사용자의 건강 정보를 선언해줍니다.
2. 요리 레시피에 관한 정보를 선언해줍니다.
3. 이 챗봇의 정확한 목적(1가지)만 정해줍니다
4. 출력될 특정 형식을 정해줍니다

- etc: context,question

  




# FAQ

## 프롬포트는 어떻게 작성해야할까요?

- 내가 사용하는 모델에 따라서 차이를 줘야합니다. 모델의 복잡성과 언어를 처리하는 능력은 상의하므로, 주로 GPTs와 같이 거대한 모델들은 복잡한 프롬포트도 잘 이해하고 받아드립니다. 그러나 학습된 언어가 다르거나, 모델의 물리적인 사이즈가 작고, 한계가 명확히 존재한다면 프롬포트도 이에 따라서 간결해질 필요가 있습니다.

---

## 프롬포트 작성 시 유의해야할 사항이 있을까요?

- 코드에서 프롬포트를 작성할때 주로 실수하는 것은 작성자의 기준대로 프롬포트를 작성한다는 것입니다. 파일의 경로를 지정한다는지, 코드 내에서 실행되는 무엇을 특정지어 이해하라고 한다면 모델은 프롬포트를 전혀 이해할 수 없습니다.
- 모델의 한계를 명확하게 파악해야합니다. 주로 로컬에서 돌려지는 모델은 chat GPT 또는 clode3.5와의 성능을 비교하면 현저히 성능이 나쁜 모델입니다. 그러므로 이 모델의 기본 지식이 얼마나 있는지, 어느정도 수준에서 환각이 이루어지는지 파악을 한다면 어느 수준의 프롬포트를 작성해야할지 알 수 있습니다. 이는 Rag 과 prompt를 결합하여 실행 하였을때 온전히 원하는대로 지식을 주고받지 못하는 이유이기도 합니다.
- 중복된 정보를 입력하는것은 혼동을 줄 수 있습니다. 의미적으로나 명시적으로 중복된 목적, 또는 정보는 강조가 된다고 생각할 수 있는데 그러지 않습니다. 

---

## 프롬포트 작성 시 가장 중요하게 생각해야할 것이 있을까요?
- 한가지의 목적: 이 챗봇의 궁극적인 목적을 지정해주는 것이 프롬포트에서 가장 중요한 것 이라고 생각합니다.
- 변수로 만들자: 내가 챗봇에게 알려주고 싶은 정보를 markdown 형식 또는 python format 문법대로 작성하여 변수로 만들어 준 뒤 프롬포트 내에서 선언을 해주는 것이 좋습니다. 특정적인 부분을 더욱 세분화 하여 받아드릴수 있습니다.(코드 내에선 두개의 문법이 혼용되었지만, 통일 시키는 것이 좋습니다.)
- 형식: 출력될 형식을 추상적으로 지정해주는 것이 좋습니다. 왜 형식의 틀을 정확하게 주지 않고 추상적으로 주는지에 대해 의문이 들 수 있습니다. 이것에 대한 이유는 우리는 보통 챗봇을 이용할떄도 주로 추상적이게 질문하고 형식적인 답변을 원합니다. 그러므로 틀에 맞춰진 형식을 프롬포트에 제공하는 것은 챗봇이 답변 과정에서 잘 지켜지지 않을 가능성이 높습니다.
